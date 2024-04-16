use std::{error::Error, sync::{Arc, Mutex}};

use neat::*;
use rand::prelude::*;
use plotters::prelude::*;

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

struct PlottingNG {
    performance_stats: Arc<Mutex<Vec<PerformanceStats>>>,
}

impl NextgenFn<AgentDNA> for PlottingNG {
    fn next_gen(&self, fitness: Vec<(AgentDNA, f32)>) -> Vec<AgentDNA> {
        let l = fitness.len();

        let high = fitness[0].1;

        let median = fitness[l / 2].1;

        let low = fitness[l-1].1;

        let mut ps = self.performance_stats.lock().unwrap();
        ps.push(PerformanceStats { high, median, low });

        division_pruning_nextgen(fitness)
    }
}

struct PerformanceStats {
    high: f32,
    median: f32,
    low: f32,
}

const OUTPUT_FILE_NAME: &'static str = "fitness-plot.png";
const GENS: usize = 100;
fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = rand::thread_rng();

    let performance_stats = Arc::new(Mutex::new(Vec::with_capacity(GENS)));
    let ng = PlottingNG { performance_stats: performance_stats.clone() };

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        fitness,
        ng,
    );

    println!("Training...");
    
    for _ in 0..GENS {
        sim.next_generation();
    }

    println!("Training complete, collecting data and building chart...");

    let root = BitMapBackend::new(OUTPUT_FILE_NAME, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("agent fitness values per generation", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0usize..100, 0f32..200.0)?;

    chart.configure_mesh().draw()?;

    let data: Vec<_> = Arc::into_inner(performance_stats).unwrap().into_inner().unwrap()
        .into_iter()
        .enumerate()
        .collect();

    let highs = data
        .iter()
        .map(|(i, PerformanceStats { high, .. })| (*i, *high));

    let medians = data
        .iter()
        .map(|(i, PerformanceStats { median, .. })| (*i, *median));

    let lows = data
        .iter()
        .map(|(i, PerformanceStats { low, .. })| (*i, *low));

    chart
        .draw_series(LineSeries::new(highs, &GREEN))?
        .label("high");

    chart
        .draw_series(LineSeries::new(medians, &YELLOW))?
        .label("median");

    chart
        .draw_series(LineSeries::new(lows, &RED))?
        .label("low");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("Complete");
    
    Ok(())
}