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
		public double m = 0;
		public double s = Math.Sqrt(3);
		#endregion

		#region -- Constructor --
		public Synapse(Neuron inputNeuron, Neuron outputNeuron)
		{
			InputNeuron = inputNeuron;
			OutputNeuron = outputNeuron;
			Weight = Network.GetRandom(m,s);
			Console.WriteLine($"Weight:{Weight}");
		}
		#endregion
	}
}