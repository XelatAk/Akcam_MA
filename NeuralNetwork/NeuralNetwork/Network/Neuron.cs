﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Network
{
	public class Neuron
	{
		#region -- Properties --
		public List<Synapse> InputSynapses { get; set; }
		public List<Synapse> OutputSynapses { get; set; }
		public double Bias { get; set; }
		public double BiasDelta { get; set; }
		public double Gradient { get; set; }
		public double Value { get; set; }
		public double min = -0.2;
		public double max = 0.2;
		
		#endregion

		#region -- Constructors --
		public Neuron()
		{
			InputSynapses = new List<Synapse>();
			OutputSynapses = new List<Synapse>();
			Bias = Network.GetRandom(min,max);
			Console.WriteLine($"Bias:{Bias}");
		}

		public Neuron(IEnumerable<Neuron> inputNeurons) : this()
		{
			foreach (var inputNeuron in inputNeurons)
			{
				var synapse = new Synapse(inputNeuron, this);
				inputNeuron.OutputSynapses.Add(synapse);
				InputSynapses.Add(synapse);
			}
		}
		#endregion

		#region -- Values & Weights --
		public virtual double CalculateValue()
		{
			return Value = Sigmoid.Output(InputSynapses.Sum(a => a.Weight * a.InputNeuron.Value) + Bias);
			
		}
		
		public double CalculateError(double target)
		{
			return Math.Abs(target - Value);
		}
				
		public double CalculateGradient(double? target = null)
		{
			if (target == null)
				return Gradient = OutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) * Sigmoid.Derivative(Value);

			else //edit
				return Gradient = CalculateError(target.Value) * Sigmoid.Derivative(Value);
		}

		public void UpdateWeights(double learnRate, double momentum)
		{
			var prevDelta = BiasDelta;
			BiasDelta = learnRate * Gradient;
			Bias += BiasDelta + momentum * prevDelta;

			foreach (var synapse in InputSynapses)
			{
				prevDelta = synapse.WeightDelta;
				synapse.WeightDelta = learnRate * Gradient * synapse.InputNeuron.Value;
				synapse.Weight += synapse.WeightDelta + momentum * prevDelta;
			}
			
		}
		#endregion
	}
}