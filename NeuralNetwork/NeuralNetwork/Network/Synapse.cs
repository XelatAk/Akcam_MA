using System;

namespace NeuralNetwork.Network
{
	public class Synapse
	{
		#region -- Properties --
		public Neuron InputNeuron { get; set; }
		public Neuron OutputNeuron { get; set; }
		public double Weight { get; set; }
		public double WeightDelta { get; set; }
		public double min = -0.2;
		public double max = 0.2;		
		#endregion

		#region -- Constructor --
		public Synapse(Neuron inputNeuron, Neuron outputNeuron)
		{
			InputNeuron = inputNeuron;
			OutputNeuron = outputNeuron;
			Weight = Network.GetRandom(min,max);
			Console.WriteLine($"Weight:{Weight}");
		}
		#endregion
	}
}