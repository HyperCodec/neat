use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use indicatif::{ProgressBar, ProgressStyle};
use neat::*;
use plotters::prelude::*;
use rand::prelude::*;

#[derive(RandomlyMutable, DivisionReproduction, Clone)]
struct AgentDNA {
    network: NeuralNetworkTopology<2, 1>,
}

impl Prunable for AgentDNA {}

impl GenerateRandom for AgentDNA {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self {
            network: NeuralNetworkTopology::new(0.01, 3, rng),
        }
    }
}

fn fitness(g: &AgentDNA) -> f32 {
    let network = NeuralNetwork::from(&g.network);
    let mut fitness = 0.;
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let n = rng.gen::<f32>() * 10000.;
        let base = rng.gen::<f32>() * 10.;
        let expected = n.log(base);

        let [answer] = network.predict([n, base]);
        network.flush_state();

        fitness += 5. / (answer - expected).abs();
    }

    fitness
}

struct PlottingNG<F: NextgenFn<AgentDNA>> {
    performance_stats: Arc<Mutex<Vec<PerformanceStats>>>,
    actual_ng: F,
}

impl<F: NextgenFn<AgentDNA>> NextgenFn<AgentDNA> for PlottingNG<F> {
    fn next_gen(&self, mut fitness: Vec<(AgentDNA, f32)>) -> Vec<AgentDNA> {
        // it's a bit slower because of sorting twice but I don't want to rewrite the nextgen.
        fitness.sort_by(|(_, fa), (_, fb)| fa.partial_cmp(fb).unwrap());

        let l = fitness.len();

        let high = fitness[l - 1].1;

        let median = fitness[l / 2].1;

        let low = fitness[0].1;

        let mut ps = self.performance_stats.lock().unwrap();
        ps.push(PerformanceStats { high, median, low });

        self.actual_ng.next_gen(fitness)
    }
}

struct PerformanceStats {
    high: f32,
    median: f32,
    low: f32,
}

const OUTPUT_FILE_NAME: &'static str = "fitness-plot.svg";
const GENS: usize = 1000;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "rayon"))]
    let mut rng = rand::thread_rng();

    let performance_stats = Arc::new(Mutex::new(Vec::with_capacity(GENS)));
    let ng = PlottingNG {
        performance_stats: performance_stats.clone(),
        actual_ng: division_pruning_nextgen,
    };

    let mut sim = GeneticSim::new(
        #[cfg(not(feature = "rayon"))]
        Vec::gen_random(&mut rng, 100),
        #[cfg(feature = "rayon")]
        Vec::gen_random(100),
        fitness,
        ng,
    );

    let pb = ProgressBar::new(GENS as u64)
        .with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} | {msg} {pos}/{len}",
            )
            .unwrap(),
        )
        .with_message("gen");

    println!("Training...");

    for _ in 0..GENS {
        sim.next_generation();

        pb.inc(1);
    }

    pb.finish();

    // prevent `Arc::into_inner` from failing
    drop(sim);

    println!("Training complete, collecting data and building chart...");

    let data: Vec<_> = Arc::into_inner(performance_stats)
        .unwrap()
        .into_inner()
        .unwrap()
        .into_iter()
        .enumerate()
        .collect();

    let highs: Vec<_> = data
        .iter()
        .map(|(i, PerformanceStats { high, .. })| (*i, *high))
        .collect();
    
    let highest_overall = highs
        .iter()
        .map(|(_, h)| h)
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let medians = data
        .iter()
        .map(|(i, PerformanceStats { median, .. })| (*i, *median));

    let lows = data
        .iter()
        .map(|(i, PerformanceStats { low, .. })| (*i, *low));

    let root = SVGBackend::new(OUTPUT_FILE_NAME, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "agent fitness values per generation",
            ("sans-serif", 50).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0usize..GENS, 0f32..*highest_overall)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(highs, &GREEN))?
        .label("high");

    chart
        .draw_series(LineSeries::new(medians, &YELLOW))?
        .label("median");

    chart.draw_series(LineSeries::new(lows, &RED))?.label("low");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("Complete");

    Ok(())
}
