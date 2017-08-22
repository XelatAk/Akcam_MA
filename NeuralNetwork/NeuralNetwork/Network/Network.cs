using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Network
{
	public class Network
	{
		#region -- Properties --
		public double LearnRate_Output { get; set; }
		public double LearnRate_Hidden { get; set; }
		public double Momentum { get; set; }
		public List<Neuron> InputLayer { get; set; }
		public List<Neuron> HiddenLayer { get; set; }
		public List<Neuron> OutputLayer { get; set; }
		#endregion

		#region -- Globals --
		private static readonly Random Random = new Random();
		#endregion

		#region -- Constructor --
		public Network(int inputSize, int hiddenSize, int outputSize)
		{
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
				Console.WriteLine(numEpochs);
			}
		}

		public void Train(List<DataSet> dataSets, double minimumError)
		{
			var error = 2.0;
			var numEpochs = 0;

			while (error > minimumError && numEpochs < int.MaxValue)
			{
				var errors = new List<double>();

				if (10 < numEpochs && numEpochs < 21)
				{
					LearnRate_Hidden = 0.4;
					LearnRate_Output = 0.2;
					Momentum = 0.2;
				}
				else if (20 < numEpochs && numEpochs < 41)
				{
					LearnRate_Hidden = 0.2;
					LearnRate_Output = 0.15;
					Momentum = 0.05;
				}
				else if (40 < numEpochs && numEpochs < 100)
				{
					LearnRate_Hidden = 0.1;
					LearnRate_Output = 0.05;
					Momentum = 0.025;
				}
				else if(99 < numEpochs)
				{
					LearnRate_Hidden = 0.05;
					LearnRate_Output = 0.025;
					Momentum = 0.01;
				}
				else
				{
					LearnRate_Hidden = 0.6;
					LearnRate_Output = 0.3;
					Momentum = 0.4;
				}

				foreach (var dataSet in dataSets)
				{
					ForwardPropagate(dataSet.Values);
					BackPropagate(dataSet.Targets);
					errors.Add(CalculateError(dataSet.Targets));
				}
				error = errors.Average();
				Console.WriteLine($"Average Error:{error}");
				
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
			HiddenLayer.ForEach(a => a.UpdateWeights(LearnRate_Hidden, Momentum));
			OutputLayer.ForEach(a => a.UpdateWeights(LearnRate_Output, Momentum));
			
		}

		public double[] Compute(params double[] inputs)
		{
			ForwardPropagate(inputs);
			return OutputLayer.Select(a => a.Value).ToArray();
		}

		private double CalculateError(params double[] targets)  //edit
		{
			var i = 0;
			return OutputLayer.Sum(a => a.CalculateError(targets[i++]));
		}

		#endregion

		#region -- Helpers --
		#region -- Normal Distribution--
		////public static double GetRandom()
		////{
		////	return 2 * Random.NextDouble() - 1;
		////	//return 2 * Random.NextDouble() - 1 < 0 ? -1 : 1;

		////}

		///// <summary>
		///// Compute a Gaussian random number.
		///// </summary>
		///// <param name="m">The mean.</param>
		///// <param name="s">The standard deviation.</param>
		///// <returns>The random number.</returns>
		//public static double GetRandom(double m, double s)
		//{
		//	double x1, x2, w, y1, y2=0;
		//	bool useLast = false;

		//	// Use value from previous call
		//	if (useLast)
		//	{
		//		y1 = y2;
		//		useLast = false;
		//	}
		//	else
		//	{

		//		do
		//		{
		//			x1 = 2.0d * Random.NextDouble() - 1.0d;    // NextDouble() is uniform in 0..1
		//			x2 = 2.0d * Random.NextDouble() - 1.0d;
		//			w = x1 * x1 + x2 * x2;
		//		} while (w >= 1.0d);

		//		w = Math.Sqrt((-2.0d * Math.Log(w)) / w);
		//		y1 = x1 * w;
		//		y2 = x2 * w;
		//		useLast = true;
		//	}

		//	return (m + y1 * s);
		//	}
		#endregion

		#region --Uniform Distribution--
		public static double GetRandom(double min, double max)
		{
			double dbl = Random.NextDouble() * (max - min) + min;
			Console.WriteLine(dbl.ToString());
			return 0;
		}
		#endregion

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