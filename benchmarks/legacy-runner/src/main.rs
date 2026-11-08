use std::{env, hint::black_box, process, time::Instant};

#[cfg(feature = "mpi")]
use std::num::NonZeroUsize;

const BASELINE_COMMIT: &str = "34cb1b1371c8b2f8ef750e2d49d10f9ef8f0782e";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Case {
    Map,
    MapIn,
    Pmap,
    Broadcast,
    Scatter,
    Gather,
}

impl Case {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "map" => Ok(Self::Map),
            "map-in" => Ok(Self::MapIn),
            "pmap" => Ok(Self::Pmap),
            "broadcast" => Ok(Self::Broadcast),
            "scatter" => Ok(Self::Scatter),
            "gather" => Ok(Self::Gather),
            _ => Err(format!("unknown case: {value}")),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Map => "map",
            Self::MapIn => "map-in",
            Self::Pmap => "pmap",
            Self::Broadcast => "broadcast",
            Self::Scatter => "scatter",
            Self::Gather => "gather",
        }
    }

    const fn is_mpi(self) -> bool {
        matches!(
            self,
            Self::Pmap | Self::Broadcast | Self::Scatter | Self::Gather
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Sequential,
    Outer,
    Inner,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "sequential" => Ok(Self::Sequential),
            "outer" => Ok(Self::Outer),
            "inner" => Ok(Self::Inner),
            _ => Err(format!("invalid mode: {value}")),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Outer => "outer",
            Self::Inner => "inner",
        }
    }

    #[cfg(feature = "rayon")]
    const fn local_mode(self) -> hataori::LocalMode {
        match self {
            Self::Sequential => hataori::LocalMode::Sequential,
            Self::Outer => hataori::LocalMode::Outer,
            Self::Inner => hataori::LocalMode::Inner,
        }
    }
}

#[derive(Debug)]
struct Config {
    case: Case,
    items: usize,
    payload_bytes: usize,
    work: u64,
    repetitions: usize,
    warmups: usize,
    threads: usize,
    mode: Mode,
    batch_size: usize,
    prefetch: bool,
}

fn positive(name: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid {name}: {value}"))?;
    if parsed == 0 {
        Err(format!("{name} must be positive"))
    } else {
        Ok(parsed)
    }
}

fn positive_u64(name: &str, value: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("invalid {name}: {value}"))?;
    if parsed == 0 {
        Err(format!("{name} must be positive"))
    } else {
        Ok(parsed)
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, String> {
    let mut args = args.into_iter();
    let case = Case::parse(&args.next().ok_or("missing CASE")?)?;
    let mut items = None;
    let mut payload_bytes = None;
    let mut work = None;
    let mut repetitions = None;
    let mut warmups = None;
    let mut threads = None;
    let mut mode = None;
    let mut batch_size = None;
    let mut prefetch = None;

    while let Some(option) = args.next() {
        let value = next_value(&mut args, &option)?;
        let duplicate = |name: &str| Err(format!("duplicate option: {name}"));
        match option.as_str() {
            "--items" => {
                if items.is_some() {
                    return duplicate("--items");
                }
                items = Some(positive("items", &value)?);
            }
            "--payload-bytes" => {
                if payload_bytes.is_some() {
                    return duplicate("--payload-bytes");
                }
                payload_bytes = Some(positive("payload-bytes", &value)?);
            }
            "--work" => {
                if work.is_some() {
                    return duplicate("--work");
                }
                work = Some(positive_u64("work", &value)?);
            }
            "--repetitions" => {
                if repetitions.is_some() {
                    return duplicate("--repetitions");
                }
                repetitions = Some(positive("repetitions", &value)?);
            }
            "--warmups" => {
                if warmups.is_some() {
                    return duplicate("--warmups");
                }
                warmups = Some(positive("warmups", &value)?);
            }
            "--threads" => {
                if threads.is_some() {
                    return duplicate("--threads");
                }
                threads = Some(positive("threads", &value)?);
            }
            "--mode" => {
                if mode.is_some() {
                    return duplicate("--mode");
                }
                mode = Some(Mode::parse(&value)?);
            }
            "--batch-size" => {
                if batch_size.is_some() {
                    return duplicate("--batch-size");
                }
                batch_size = Some(positive("batch-size", &value)?);
            }
            "--prefetch" => {
                if prefetch.is_some() {
                    return duplicate("--prefetch");
                }
                prefetch = Some(match value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err(format!("invalid prefetch: {value}")),
                });
            }
            _ => return Err(format!("unknown option: {option}")),
        }
    }

    if (threads.is_some() || mode.is_some()) && !matches!(case, Case::MapIn | Case::Pmap) {
        return Err("--threads/--mode require map-in or pmap".into());
    }
    if (batch_size.is_some() || prefetch.is_some()) && case != Case::Pmap {
        return Err("--batch-size/--prefetch require pmap".into());
    }
    if case == Case::MapIn && !cfg!(feature = "rayon") {
        return Err("map-in requires the rayon feature".into());
    }
    if case.is_mpi() && !cfg!(feature = "mpi") {
        return Err(format!("{} requires the mpi feature", case.name()));
    }

    let config = Config {
        case,
        items: items.unwrap_or(8),
        payload_bytes: payload_bytes.unwrap_or(16),
        work: work.unwrap_or(16),
        repetitions: repetitions.unwrap_or(1),
        warmups: warmups.unwrap_or(1),
        threads: threads.unwrap_or(1),
        mode: mode.unwrap_or(Mode::Sequential),
        batch_size: batch_size.unwrap_or(1),
        prefetch: prefetch.unwrap_or(false),
    };
    if config.case == Case::Pmap
        && !cfg!(feature = "rayon")
        && (config.mode != Mode::Sequential || config.prefetch)
    {
        return Err("MPI-only pmap requires sequential mode and prefetch=false".into());
    }
    Ok(config)
}

#[cfg_attr(feature = "mpi", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Eq, PartialEq)]
struct Item {
    index: u64,
    payload: Vec<u8>,
}

fn make_items(count: usize, payload_bytes: usize, rank: u64) -> Vec<Item> {
    (0..count)
        .map(|index| Item {
            index: index as u64,
            payload: (0..payload_bytes)
                .map(|offset| (index as u64 ^ rank ^ offset as u64).wrapping_mul(31) as u8)
                .collect(),
        })
        .collect()
}

fn transform(mut item: Item, work: u64) -> Item {
    let mut value = black_box(item.index ^ 0x9e37_79b9_7f4a_7c15);
    for step in 0..work {
        value = value.rotate_left(13).wrapping_mul(0xbf58_476d_1ce4_e5b9)
            ^ black_box(step.wrapping_add(item.index));
    }
    for (offset, byte) in item.payload.iter_mut().enumerate() {
        *byte ^= value.rotate_left((offset % 64) as u32) as u8;
    }
    black_box(item)
}

fn checksum(items: &[Item]) -> u64 {
    items.iter().fold(0xcbf2_9ce4_8422_2325, |hash, item| {
        item.payload.iter().fold(
            hash.rotate_left(7) ^ item.index.wrapping_mul(0x9e37_79b9),
            |hash, byte| hash.wrapping_mul(0x100_0000_01b3) ^ u64::from(*byte),
        )
    })
}

fn expected(config: &Config, count: usize, rank: u64) -> (u64, usize) {
    let output: Vec<_> = make_items(count, config.payload_bytes, rank)
        .into_iter()
        .map(|item| transform(item, config.work))
        .collect();
    (
        checksum(&output),
        output.iter().map(|item| item.payload.len()).sum(),
    )
}

struct Record<'a> {
    config: &'a Config,
    repetition: usize,
    elapsed_ns: u128,
    checksum: u64,
    items: usize,
    ranks: i32,
}

fn print_record(record: Record<'_>) {
    let config = record.config;
    println!(
        "HATAORI_BENCH\tcase={}\trepetition={}\telapsed_ns={}\tchecksum={}\titems={}\tpayload_bytes={}\twork={}\tranks={}\tthreads={}\tmode={}\tbatch_size={}\tprefetch={}\twarmups={}\tbaseline_commit={}",
        config.case.name(),
        record.repetition,
        record.elapsed_ns,
        record.checksum,
        record.items,
        config.payload_bytes,
        config.work,
        record.ranks,
        config.threads,
        config.mode.name(),
        config.batch_size,
        config.prefetch,
        config.warmups,
        BASELINE_COMMIT,
    );
}

fn validate(
    output: &[Item],
    expected_checksum: u64,
    expected_items: usize,
    expected_bytes: usize,
) -> Result<u64, String> {
    let actual_checksum = checksum(output);
    let actual_bytes: usize = output.iter().map(|item| item.payload.len()).sum();
    if output.len() != expected_items
        || actual_bytes != expected_bytes
        || actual_checksum != expected_checksum
    {
        return Err(format!(
            "result validation failed: items {}/{expected_items}, bytes {actual_bytes}/{expected_bytes}, checksum {actual_checksum}/{expected_checksum}",
            output.len()
        ));
    }
    Ok(actual_checksum)
}

fn run_map(config: &Config) -> Result<(), String> {
    let (expected_checksum, expected_bytes) = expected(config, config.items, 0);
    for iteration in 0..config.warmups + config.repetitions {
        let input = make_items(config.items, config.payload_bytes, 0);
        let start = Instant::now();
        let output = hataori::map(input, |item| Ok::<_, String>(transform(item, config.work)))
            .map_err(|error| error.to_string())?;
        let elapsed_ns = start.elapsed().as_nanos();
        let actual = validate(&output, expected_checksum, config.items, expected_bytes)?;
        if iteration >= config.warmups {
            print_record(Record {
                config,
                repetition: iteration - config.warmups,
                elapsed_ns,
                checksum: actual,
                items: config.items,
                ranks: 1,
            });
        }
    }
    Ok(())
}

#[cfg(feature = "rayon")]
fn managed_domain(threads: usize) -> Result<hataori::Domain, String> {
    // The pinned baseline rejects a CPU id outside this process's affinity
    // mask, and a runner may reserve cpu 0, so the ids come from the mask
    // instead of `0..threads`. On a host that may use `0..threads` this is the
    // same set; on a narrowed mask it is the only set the baseline accepts.
    hataori::Domain::managed(allowed_cpus(threads)?, threads).map_err(|error| error.to_string())
}

#[cfg(all(feature = "rayon", target_os = "linux"))]
fn allowed_cpus(threads: usize) -> Result<Vec<usize>, String> {
    let mut allowed = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let result =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut allowed) };
    if result != 0 {
        return Err(format!(
            "affinity query failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let cpus: Vec<usize> = (0..libc::CPU_SETSIZE as usize)
        .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &allowed) })
        .take(threads)
        .collect();
    if cpus.len() < threads {
        return Err(format!(
            "{threads} threads requested, but only {} CPUs are available to this process",
            cpus.len()
        ));
    }
    Ok(cpus)
}

#[cfg(all(feature = "rayon", not(target_os = "linux")))]
fn allowed_cpus(threads: usize) -> Result<Vec<usize>, String> {
    Ok((0..threads).collect())
}

#[cfg(all(feature = "mpi", feature = "rayon"))]
fn hybrid_domain(threads: usize) -> Result<hataori::Domain, String> {
    use std::sync::Arc;

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| error.to_string())?,
    );
    hataori::Domain::external(pool, (0..threads).collect(), threads)
        .map_err(|error| error.to_string())
}

#[cfg(feature = "rayon")]
fn run_map_in(config: &Config) -> Result<(), String> {
    let domain = managed_domain(config.threads)?;
    let (expected_checksum, expected_bytes) = expected(config, config.items, 0);
    for iteration in 0..config.warmups + config.repetitions {
        let input = make_items(config.items, config.payload_bytes, 0);
        let start = Instant::now();
        let output = hataori::map_in(&domain, config.mode.local_mode(), input, |item| {
            Ok::<_, String>(transform(item, config.work))
        })
        .map_err(|error| error.to_string())?;
        let elapsed_ns = start.elapsed().as_nanos();
        let actual = validate(&output, expected_checksum, config.items, expected_bytes)?;
        if iteration >= config.warmups {
            print_record(Record {
                config,
                repetition: iteration - config.warmups,
                elapsed_ns,
                checksum: actual,
                items: config.items,
                ranks: 1,
            });
        }
    }
    Ok(())
}

#[cfg(not(feature = "rayon"))]
fn run_map_in(_: &Config) -> Result<(), String> {
    Err("map-in requires the rayon feature".into())
}

#[cfg(feature = "mpi")]
fn globally_valid<C: mpi::traits::CommunicatorCollectives>(world: &C, valid: bool) -> bool {
    use mpi::collective::SystemOperation;
    let local = i32::from(valid);
    let mut global = 0;
    world.all_reduce_into(&local, &mut global, SystemOperation::min());
    global == 1
}

#[cfg(all(feature = "mpi", not(feature = "rayon")))]
fn run_pmap<C: mpi::traits::Communicator>(world: &C, config: &Config) -> Result<(), String> {
    let rank = world.rank();
    let (expected_checksum, expected_bytes) = expected(config, config.items, 0);
    let domain = hataori::Domain::sequential();
    for iteration in 0..config.warmups + config.repetitions {
        let input = (rank == 0).then(|| make_items(config.items, config.payload_bytes, 0));
        let start = Instant::now();
        let output = hataori::pmap(
            world,
            &domain,
            hataori::PmapOptions {
                batch_size: NonZeroUsize::new(config.batch_size).unwrap(),
                ..hataori::PmapOptions::default()
            },
            input,
            |item| Ok::<_, String>(transform(item, config.work)),
        )
        .map_err(|error| error.to_string())?;
        let elapsed_ns = start.elapsed().as_nanos();
        let valid = output.as_ref().is_none_or(|output| {
            validate(output, expected_checksum, config.items, expected_bytes).is_ok()
        });
        if !globally_valid(world, valid) {
            return Err("pmap result validation failed on at least one rank".into());
        }
        if rank == 0 && iteration >= config.warmups {
            let output = output.ok_or("rank zero did not receive pmap output")?;
            print_record(Record {
                config,
                repetition: iteration - config.warmups,
                elapsed_ns,
                checksum: checksum(&output),
                items: output.len(),
                ranks: world.size(),
            });
        }
    }
    Ok(())
}

#[cfg(all(feature = "mpi", feature = "rayon"))]
fn run_pmap<C: mpi::traits::Communicator>(world: &C, config: &Config) -> Result<(), String> {
    let rank = world.rank();
    let (expected_checksum, expected_bytes) = expected(config, config.items, 0);
    let domain = hybrid_domain(config.threads)?;
    for iteration in 0..config.warmups + config.repetitions {
        let input = (rank == 0).then(|| make_items(config.items, config.payload_bytes, 0));
        let start = Instant::now();
        let output = hataori::pmap(
            world,
            &domain,
            hataori::PmapOptions {
                batch_size: NonZeroUsize::new(config.batch_size).unwrap(),
                local_mode: config.mode.local_mode(),
                prefetch: config.prefetch,
                ..hataori::PmapOptions::default()
            },
            input,
            |item| Ok::<_, String>(transform(item, config.work)),
        )
        .map_err(|error| error.to_string())?;
        let elapsed_ns = start.elapsed().as_nanos();
        let valid = output.as_ref().is_none_or(|output| {
            validate(output, expected_checksum, config.items, expected_bytes).is_ok()
        });
        if !globally_valid(world, valid) {
            return Err("pmap result validation failed on at least one rank".into());
        }
        if rank == 0 && iteration >= config.warmups {
            let output = output.ok_or("rank zero did not receive pmap output")?;
            print_record(Record {
                config,
                repetition: iteration - config.warmups,
                elapsed_ns,
                checksum: checksum(&output),
                items: output.len(),
                ranks: world.size(),
            });
        }
    }
    Ok(())
}

#[cfg(feature = "mpi")]
fn run_placement<C: mpi::traits::Communicator>(world: &C, config: &Config) -> Result<(), String> {
    let rank = world.rank();
    let size = world.size();
    for iteration in 0..config.warmups + config.repetitions {
        let (elapsed_ns, output, expected_items, expected_rank) = match config.case {
            Case::Broadcast => {
                let input: Option<Vec<Item>> = (rank == 0).then(|| {
                    make_items(config.items, config.payload_bytes, 0)
                        .into_iter()
                        .map(|item| transform(item, config.work))
                        .collect()
                });
                let start = Instant::now();
                let output =
                    hataori::broadcast(world, 0, input).map_err(|error| error.to_string())?;
                (start.elapsed().as_nanos(), output, config.items, 0_u64)
            }
            Case::Scatter => {
                let input: Option<Vec<Vec<Item>>> = (rank == 0).then(|| {
                    (0..size)
                        .map(|target| {
                            make_items(config.items, config.payload_bytes, target as u64)
                                .into_iter()
                                .map(|item| transform(item, config.work))
                                .collect()
                        })
                        .collect()
                });
                let start = Instant::now();
                let output =
                    hataori::scatter(world, 0, input).map_err(|error| error.to_string())?;
                (
                    start.elapsed().as_nanos(),
                    output,
                    config.items,
                    rank as u64,
                )
            }
            Case::Gather => {
                let input: Vec<Item> = make_items(config.items, config.payload_bytes, rank as u64)
                    .into_iter()
                    .map(|item| transform(item, config.work))
                    .collect();
                let start = Instant::now();
                let gathered =
                    hataori::gather(world, 0, input).map_err(|error| error.to_string())?;
                let elapsed = start.elapsed().as_nanos();
                let output: Vec<Item> =
                    gathered.unwrap_or_default().into_iter().flatten().collect();
                (elapsed, output, config.items * size as usize, u64::MAX)
            }
            _ => return Err("internal non-placement dispatch".into()),
        };

        let valid = if config.case == Case::Gather && rank != 0 {
            output.is_empty()
        } else if expected_rank == u64::MAX {
            let expected_output: Vec<_> = (0..size)
                .flat_map(|source| {
                    make_items(config.items, config.payload_bytes, source as u64)
                        .into_iter()
                        .map(|item| transform(item, config.work))
                })
                .collect();
            output == expected_output
        } else {
            let expected_output: Vec<_> =
                make_items(config.items, config.payload_bytes, expected_rank)
                    .into_iter()
                    .map(|item| transform(item, config.work))
                    .collect();
            output == expected_output
        };
        if !globally_valid(world, valid) {
            return Err(format!("{} result validation failed", config.case.name()));
        }
        if rank == 0 && iteration >= config.warmups {
            print_record(Record {
                config,
                repetition: iteration - config.warmups,
                elapsed_ns,
                checksum: checksum(&output),
                items: expected_items,
                ranks: size,
            });
        }
    }
    Ok(())
}

#[cfg(feature = "mpi")]
fn run_mpi(config: &Config) -> Result<(), String> {
    #[cfg(feature = "rayon")]
    let universe = {
        use mpi::environment::Threading;
        let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
            .ok_or("MPI was already initialized or finalized")?;
        if provided < Threading::Funneled {
            return Err("mpi+rayon requires MPI_THREAD_FUNNELED".into());
        }
        universe
    };
    #[cfg(not(feature = "rayon"))]
    let universe = mpi::initialize().ok_or("MPI was already initialized or finalized")?;
    let world = universe.world();
    match config.case {
        Case::Pmap => run_pmap(&world, config),
        Case::Broadcast | Case::Scatter | Case::Gather => run_placement(&world, config),
        _ => Err("internal non-MPI dispatch".into()),
    }
}

#[cfg(not(feature = "mpi"))]
fn run_mpi(_: &Config) -> Result<(), String> {
    Err("MPI case requires the mpi feature".into())
}

fn run() -> Result<(), String> {
    let config = parse_args(env::args().skip(1))?;
    match config.case {
        Case::Map => run_map(&config),
        Case::MapIn => run_map_in(&config),
        _ => run_mpi(&config),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("hataori-legacy-runner: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::{checksum, make_items, parse_args, transform, Case};

    fn parse(input: &str) -> Result<super::Config, String> {
        parse_args(input.split_whitespace().map(String::from))
    }

    #[test]
    fn parser_rejects_unknown_duplicate_and_zero_options() {
        assert!(parse("map --not-an-option 1").is_err());
        assert!(parse("map --items 1 --items 2").is_err());
        assert!(parse("map --items 0").is_err());
    }

    #[test]
    fn parser_keeps_case_specific_options_narrow() {
        assert!(parse("map --threads 2").is_err());
        assert!(parse("broadcast --batch-size 2").is_err());
        assert_eq!(parse("map --items 2").unwrap().case, Case::Map);
    }

    #[test]
    fn workload_depends_on_values_and_is_repeatable() {
        let first: Vec<_> = make_items(3, 8, 0)
            .into_iter()
            .map(|item| transform(item, 5))
            .collect();
        let second: Vec<_> = make_items(3, 8, 0)
            .into_iter()
            .map(|item| transform(item, 5))
            .collect();
        assert_eq!(first, second);
        assert_ne!(checksum(&first[..1]), checksum(&first[1..2]));
    }
}
