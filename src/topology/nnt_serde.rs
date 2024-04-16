use super::*;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// A serializable wrapper for [`NeuronTopology`]. See [`NNTSerde::from`] for conversion.
#[derive(Serialize, Deserialize)]
pub struct NNTSerde<const I: usize, const O: usize> {
    #[serde(with = "BigArray")]
    pub(crate) input_layer: [NeuronTopology; I],

    pub(crate) hidden_layers: Vec<NeuronTopology>,

    #[serde(with = "BigArray")]
    pub(crate) output_layer: [NeuronTopology; O],

    pub(crate) mutation_rate: f32,
    pub(crate) mutation_passes: usize,
}

impl<const I: usize, const O: usize> From<&NeuralNetworkTopology<I, O>> for NNTSerde<I, O> {
    fn from(value: &NeuralNetworkTopology<I, O>) -> Self {
        let input_layer = value
            .input_layer
            .iter()
            .map(|n| n.read().unwrap().clone())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hidden_layers = value
            .hidden_layers
            .iter()
            .map(|n| n.read().unwrap().clone())
            .collect();

        let output_layer = value
            .output_layer
            .iter()
            .map(|n| n.read().unwrap().clone())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
            mutation_rate: value.mutation_rate,
            mutation_passes: value.mutation_passes,
        }
    }
}

#[cfg(test)]
#[test]
fn serde() {
    let mut rng = rand::thread_rng();
    let nnt = NeuralNetworkTopology::<10, 10>::new(0.1, 3, &mut rng);
    let nnts = NNTSerde::from(&nnt);

    let encoded = bincode::serialize(&nnts).unwrap();

    if let Some(_) = option_env!("TEST_CREATEFILE") {
        std::fs::write("serde-test.nn", &encoded).unwrap();
    }

    let decoded: NNTSerde<10, 10> = bincode::deserialize(&encoded).unwrap();
    let nnt2: NeuralNetworkTopology<10, 10> = decoded.into();

    dbg!(nnt, nnt2);
}
