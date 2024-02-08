use neat::*;
use rand::prelude::*;

struct AgentDNA {
    network: NeuralNetworkTopology, 
}

impl RandomlyMutable for AgentDNA {
    fn mutate(&mut self, rate: f32, rng: &mut impl Rng) {
        self.network.mutate(rate, rng);
    }
}

impl Prunable for AgentDNA {}

impl DivisionReproduction for AgentDNA {
    fn spawn_child(&self, rng: &mut impl Rng) -> Self {
        Self {
            network: self.network.spawn_child(rng),
        }
    }
}

impl GenerateRandom for AgentDNA {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self {
            network: NeuralNetworkTopology::new(2, 4, 0.01, rng),
        }
    }
}

struct Agent {
    network: NeuralNetwork,
}

impl From<&AgentDNA> for Agent {
    fn from(value: &AgentDNA) -> Self {
        Self {
            network: (&value.network).into(),
        }
    }
}

fn fitness(dna: &AgentDNA) -> f32 {
    let agent = Agent::from(dna);
    let mut fitness = 0.;
    let mut rng = rand::thread_rng();

    for _ in 0..10 {
        // 10 games

        // set up game
        let mut agent_pos: (i32, i32) = (rng.gen_range(0..10), rng.gen_range(0..10));
        let mut food_pos: (i32, i32) = (rng.gen_range(0..10), rng.gen_range(0..10));

        while food_pos == agent_pos {
            food_pos = (rng.gen_range(0..10), rng.gen_range(0..10));
        }

        let mut step = 0;

        loop {
            // perform actions in game
            let action = agent.network.predict(vec![(food_pos.0 - agent_pos.0) as f32, (food_pos.1 - agent_pos.1) as f32]);
            let action = action.iter().max_index();

            match action {
                0 => agent_pos.0 += 1,
                1 => agent_pos.0 -= 1,
                2 => agent_pos.1 += 1,
                _ => agent_pos.1 -= 1,
            }

            step += 1;

            if agent_pos == food_pos {
                fitness += 10.;
                break; // new game
            } else {
                // lose fitness for being slow and far away
                fitness -= (food_pos.0 - agent_pos.0 + food_pos.1 - agent_pos.1).abs() as f32 * 0.001;
            }

            // 50 steps per game
            if step == 50 {
                break;
            }
        }
    }


    fitness
}

fn main() {
    let mut rng = rand::thread_rng();


}