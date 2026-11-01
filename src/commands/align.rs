//! `evnx align`: give `.env` files the layout of the reference template —
//! its sections, comments and key order — keeping every value they hold.
//!
//! The layout rules, and the comment rule, are in [`crate::core::layout`]. This
//! file reads the files, refuses what must not be aligned, reports, and writes.
//!
//! ⛔ **Names, lines and statuses only.** Nothing printed here — the report,
//! `--format json`, or an error — holds a value, a comment or a source line.
//!
//! # Order of work
//!
//! 1. The reference is read; a problem in it stops the run.
//! 2. Every target is checked against the reference and the others: the same
//!    file under two names (a symlink or a hard link) is refused.
//! 3. Every target is read and aligned in memory. A problem in **any** target
//!    stops the run before **any** file is written.
//! 4. `--check` and `--dry-run` report and stop. Otherwise each file that
//!    changes is staged (a temp file beside it), then each is renamed over its
//!    original. A file already aligned is not opened for writing.
//!
//! ⚠️ Each replacement is atomic; the set is not. If renaming the second of two
//! files fails, the first has already been replaced, and the error says which
//! files were written and which were not.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::Path;

use crate::core::layout::{LayoutError, Plan, Reference, EXTRAS_HEADING};
use crate::core::safe_write;
use crate::core::verdict::Verdict;
use crate::utils::path::label;
use crate::utils::ui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Write the files that change.
    Write,
    /// Report what would change; write nothing.
    DryRun,
    /// Report, write nothing, and exit 1 when a file is out of alignment.
    Check,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub reference: String,
    pub targets: Vec<String>,
    pub mode: Mode,
    pub json: bool,
}

/// A target that has been read and aligned in memory.
struct Prepared {
    path: String,
    original: Vec<u8>,
    plan: Plan,
}

/// Run `evnx align`. `Ok` is the verdict; `Err` is "no verdict" (exit 2).
pub fn run(opts: Options) -> Result<Verdict> {
    if opts.targets.is_empty() {
        bail!("no file to align: pass --env (or --env-name)");
    }

    let reference = read_reference(&opts.reference)?;
    refuse_aliases(&opts.reference, &opts.targets)?;

    // ── Preflight: every target, before anything is written ──────────────────
    let mut prepared = Vec::with_capacity(opts.targets.len());
    let mut problems: Vec<String> = Vec::new();
    for path in &opts.targets {
        match prepare(&reference, path) {
            Ok(p) => prepared.push(p),
            Err(lines) => problems.extend(lines),
        }
    }
    if !problems.is_empty() {
        bail!(
            "nothing was aligned and no file was written:\n{}",
            problems
                .iter()
                .map(|p| format!("  {p}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    let drifted = prepared.iter().filter(|p| p.plan.changed).count();
    match opts.mode {
        Mode::Check | Mode::DryRun => {
            report(&opts, &prepared, &[]);
            let verdict = if opts.mode == Mode::Check && drifted > 0 {
                Verdict::Findings
            } else {
                Verdict::Clean
            };
            Ok(verdict)
        }
        Mode::Write => {
            let written = write_all(&prepared)?;
            report(&opts, &prepared, &written);
            Ok(Verdict::Clean)
        }
    }
}

fn read_text(path: &str) -> std::result::Result<String, String> {
    let shown = label(path);
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!("{shown} does not exist"))
        }
        Err(e) => return Err(format!("{shown}: could not be read ({e})")),
    };
    if !meta.is_file() {
        return Err(format!("{shown} is not a regular file"));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{shown}: could not be read ({e})"))?;
    String::from_utf8(bytes).map_err(|_| format!("{shown}: {}", LayoutError::NotUtf8))
}

fn read_reference(path: &str) -> Result<Reference> {
    let text = read_text(path).map_err(|e| anyhow::anyhow!("reference {e}"))?;
    Reference::parse(&text).map_err(|errors| {
        anyhow::anyhow!(
            "the reference {} cannot be used as a layout; no file was written:\n{}",
            label(path),
            indent(&label(path), &errors)
        )
    })
}

fn indent(file: &str, errors: &[LayoutError]) -> String {
    errors
        .iter()
        .map(|e| format!("  {file}: {e}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The reference and the targets are distinct files, and so are the targets:
/// by path, through a symlink, and by hard link (same device and inode).
fn refuse_aliases(reference: &str, targets: &[String]) -> Result<()> {
    for (i, t) in targets.iter().enumerate() {
        if safe_write::same_file(Path::new(reference), Path::new(t)) {
            bail!(
                "{} is the same file as the reference {} (by path, symlink or hard link). \
                 Nothing was written.",
                label(t),
                label(reference)
            );
        }
        for earlier in &targets[..i] {
            if safe_write::same_file(Path::new(earlier), Path::new(t)) {
                bail!(
                    "{} and {} are the same file (by path, symlink or hard link); \
                     name it once. Nothing was written.",
                    label(earlier),
                    label(t)
                );
            }
        }
    }
    Ok(())
}

fn prepare(reference: &Reference, path: &str) -> std::result::Result<Prepared, Vec<String>> {
    let text = read_text(path).map_err(|e| vec![e])?;
    let plan = reference.align(&text).map_err(|errors| {
        errors
            .iter()
            .map(|e| format!("{}: {e}", label(path)))
            .collect::<Vec<_>>()
    })?;
    Ok(Prepared {
        path: path.to_string(),
        original: text.into_bytes(),
        plan,
    })
}

/// Stage every change, then commit them in order. Returns the paths written.
fn write_all(prepared: &[Prepared]) -> Result<Vec<String>> {
    let changing: Vec<&Prepared> = prepared.iter().filter(|p| p.plan.changed).collect();
    let mut staged = Vec::with_capacity(changing.len());
    for p in &changing {
        let s =
            safe_write::stage_private(&p.path, p.plan.output.as_bytes()).with_context(|| {
                format!("no file was written: could not prepare {}", label(&p.path))
            })?;
        staged.push((p, s));
    }

    let mut written: Vec<String> = Vec::new();
    for (p, s) in staged {
        let result = unchanged_since_read(p).and_then(|()| s.commit());
        match result {
            Ok(w) => {
                safe_write::report(&w);
                written.push(p.path.clone());
            }
            Err(e) => {
                let rest: Vec<String> = changing
                    .iter()
                    .map(|q| label(&q.path))
                    .filter(|q| !written.iter().any(|w| label(w) == *q))
                    .collect();
                let done = if written.is_empty() {
                    "none".to_string()
                } else {
                    written
                        .iter()
                        .map(|w| label(w))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                return Err(e.context(format!(
                    "could not replace {}. Already aligned and written: {done}. Not written: {}. \
                     Each file is replaced atomically, but the set is not one transaction.",
                    label(&p.path),
                    rest.join(", ")
                )));
            }
        }
    }
    Ok(written)
}

/// Refuse to replace a file that changed after it was read: the plan was made
/// from the old content and would overwrite the edit.
fn unchanged_since_read(p: &Prepared) -> Result<()> {
    let now = std::fs::read(&p.path).with_context(|| format!("re-reading {}", label(&p.path)))?;
    if now != p.original {
        bail!("{} changed while evnx was reading it", label(&p.path));
    }
    Ok(())
}

// ── Report ────────────────────────────────────────────────────────────────────

/// `--format json`. `schema_version` changes only when a field changes meaning.
#[derive(Serialize)]
struct JsonReport<'a> {
    schema_version: u32,
    mode: &'static str,
    reference: String,
    /// Every target already matched the reference before this run.
    aligned: bool,
    targets: Vec<JsonTarget<'a>>,
}

#[derive(Serialize)]
struct JsonTarget<'a> {
    file: String,
    /// `aligned` (nothing to do), `written`, or `out-of-alignment` (check,
    /// dry run).
    status: &'static str,
    moved: Vec<JsonMove<'a>>,
    added: Vec<JsonAdded<'a>>,
    extra: Vec<JsonMove<'a>>,
    target_only_comments: usize,
}

#[derive(Serialize)]
struct JsonMove<'a> {
    key: &'a str,
    from_line: usize,
    to_line: usize,
}

#[derive(Serialize)]
struct JsonAdded<'a> {
    key: &'a str,
    line: usize,
}

fn status(p: &Prepared, written: &[String]) -> &'static str {
    if !p.plan.changed {
        "aligned"
    } else if written.contains(&p.path) {
        "written"
    } else {
        "out-of-alignment"
    }
}

fn report(opts: &Options, prepared: &[Prepared], written: &[String]) {
    if opts.json {
        let r = JsonReport {
            schema_version: 1,
            mode: match opts.mode {
                Mode::Write => "write",
                Mode::DryRun => "dry-run",
                Mode::Check => "check",
            },
            reference: label(&opts.reference),
            aligned: prepared.iter().all(|p| !p.plan.changed),
            targets: prepared
                .iter()
                .map(|p| JsonTarget {
                    file: label(&p.path),
                    status: status(p, written),
                    moved: moves(&p.plan.moved),
                    added: p
                        .plan
                        .added
                        .iter()
                        .map(|a| JsonAdded {
                            key: &a.key,
                            line: a.line,
                        })
                        .collect(),
                    extra: moves(&p.plan.extra),
                    target_only_comments: p.plan.target_comments,
                })
                .collect(),
        };
        // Serialising names and numbers cannot fail.
        println!(
            "{}",
            serde_json::to_string_pretty(&r).expect("a report of names and numbers")
        );
        return;
    }

    let quiet = ui::is_quiet();
    for p in prepared {
        let s = match (opts.mode, p.plan.changed) {
            (_, false) => "already aligned",
            (Mode::Write, true) => "aligned",
            (Mode::DryRun, true) => "would change",
            (Mode::Check, true) => "out of alignment",
        };
        // -q: what changed or drifted still prints; "already aligned" and a
        // successful write's details do not.
        let show = match opts.mode {
            Mode::Check => p.plan.changed || !quiet,
            Mode::DryRun => true,
            Mode::Write => !quiet,
        };
        if !show {
            continue;
        }
        println!("{}  {s}", label(&p.path));
        if p.plan.changed {
            details(&p.plan);
        }
    }

    let changed = prepared.iter().filter(|p| p.plan.changed).count();
    match opts.mode {
        Mode::Write if changed > 0 => ui::success(format!(
            "Aligned {changed} of {} file{} to {}",
            prepared.len(),
            plural(prepared.len()),
            label(&opts.reference)
        )),
        Mode::Check if changed > 0 => {
            let envs: String = prepared
                .iter()
                .filter(|p| p.plan.changed)
                .map(|p| format!(" --env {}", crate::core::shell::word(&label(&p.path))))
                .collect();
            eprintln!(
                "{changed} of {} file{} out of alignment with {}. Fix: evnx align --reference {}{envs}",
                prepared.len(),
                plural(prepared.len()),
                label(&opts.reference),
                crate::core::shell::word(&label(&opts.reference)),
            );
        }
        Mode::DryRun => ui::info("Dry run: no file was written."),
        _ => {}
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

fn moves(m: &[crate::core::layout::Moved]) -> Vec<JsonMove<'_>> {
    m.iter()
        .map(|m| JsonMove {
            key: &m.key,
            from_line: m.from_line,
            to_line: m.to_line,
        })
        .collect()
}

/// One row per key: where it was and where it goes. Never a value.
fn details(plan: &Plan) {
    for m in &plan.moved {
        println!("    moved  {}  line {} → {}", m.key, m.from_line, m.to_line);
    }
    for a in &plan.added {
        println!("    added  {}  line {} (empty)", a.key, a.line);
    }
    for m in &plan.extra {
        println!(
            "    extra  {}  line {} → {}  (kept under \"{}\")",
            m.key,
            m.from_line,
            m.to_line,
            EXTRAS_HEADING.trim_start_matches("# ")
        );
    }
    if plan.target_comments > 0 {
        println!(
            "    kept   {} comment line{} found only in this file",
            plan.target_comments,
            plural(plan.target_comments)
        );
    }
    if plan.moved.is_empty() && plan.added.is_empty() && plan.extra.is_empty() {
        println!("    layout  headings, comments or blank lines from the reference");
    }
}
