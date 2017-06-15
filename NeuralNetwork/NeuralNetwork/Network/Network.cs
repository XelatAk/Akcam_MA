using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Network
{
	public class Network
	{
		#region -- Properties --
		public double LearnRate { get; set; }
		public double Momentum { get; set; }
		public List<Neuron> InputLayer { get; set; }
		public List<Neuron> HiddenLayer { get; set; }
		public List<Neuron> OutputLayer { get; set; }
		#endregion

		#region -- Globals --
		private static readonly Random Random = new Random();
		#endregion

		#region -- Constructor --
		public Network(int inputSize, int hiddenSize, int outputSize, double? learnRate = null, double? momentum = null)
		{
			LearnRate = learnRate ?? 0.4;
			Momentum = momentum ?? .3;
			InputLayer = new List<Neuron>();
			HiddenLayer = new List<Neuron>();
			OutputLayer = new List<Neuron>();

			for (var i = 0; i < inputSize; i++)
				InputLayer.Add(new Neuron());

			for (var i = 0; i < hiddenSize; i++)
				HiddenLayer.Add(new Neuron(InputLayer));

			for (var i = 0; i < outputSize; i++)
				OutputLayer.Add(new Neuron(HiddenLayer));
		}
		#endregion

		#region -- Training --
		public void Train(List<DataSet> dataSets, int numEpochs)
		{
			for (var i = 0; i < numEpochs; i++)
			{
				foreach (var dataSet in dataSets)
				{
					ForwardPropagate(dataSet.Values);
					BackPropagate(dataSet.Targets);
				}
			}
		}

		public void Train(List<DataSet> dataSets, double minimumError)
		{
			var error = 1.0;
			var numEpochs = 0;

			while (error > minimumError && numEpochs < int.MaxValue)
			{
				var errors = new List<double>();
				foreach (var dataSet in dataSets)
				{
					ForwardPropagate(dataSet.Values);
					BackPropagate(dataSet.Targets);
					errors.Add(CalculateError(dataSet.Targets));
				}
				error = errors.Average();
				Console.WriteLine(error);   //edit
				numEpochs++;
			}
		}

		private void ForwardPropagate(params double[] inputs)
		{
			var i = 0;
			InputLayer.ForEach(a => a.Value = inputs[i++]);
			HiddenLayer.ForEach(a => a.CalculateValue());
			OutputLayer.ForEach(a => a.CalculateValue());
		}

		private void BackPropagate(params double[] targets)
		{
			var i = 0;
			OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));
			HiddenLayer.ForEach(a => a.CalculateGradient());
			HiddenLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum));
			OutputLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum));
		}

		public double[] Compute(params double[] inputs)
		{
			ForwardPropagate(inputs);
			return OutputLayer.Select(a => a.Value).ToArray();
		}

		private double CalculateError(params double[] targets)
		{
			var i = 0;
			return OutputLayer.Sum(a => Math.Abs(a.CalculateError(targets[i++])));
		}
		#endregion

		#region -- Helpers --
		//public static double GetRandom()
		//{
		//	return 2 * Random.NextDouble() - 1;
		//	//return 2 * Random.NextDouble() - 1 < 0 ? -1 : 1;
			 
		//}

		/// <summary>
		/// Compute a Gaussian random number.
		/// </summary>
		/// <param name="m">The mean.</param>
		/// <param name="s">The standard deviation.</param>
		/// <returns>The random number.</returns>
		public static double GetRandom(double m, double s)
		{
			double x1, x2, w, y1, y2=0;
			bool useLast = false;

			// Use value from previous call
			if (useLast)
			{
				y1 = y2;
				useLast = false;
			}
			else
			{

				do
				{
					x1 = 2.0d * Random.NextDouble() - 1.0d;    // NextDouble() is uniform in 0..1
					x2 = 2.0d * Random.NextDouble() - 1.0d;
					w = x1 * x1 + x2 * x2;
				} while (w >= 1.0d);

				w = Math.Sqrt((-2.0d * Math.Log(w)) / w);
				y1 = x1 * w;
				y2 = x2 * w;
				useLast = true;
			}

			return (m + y1 * s);
			}
		
		#endregion
	}

	#region -- Enum --
	public enum TrainingType
	{
		Epoch,
		MinimumError
	}
	#endregion
}