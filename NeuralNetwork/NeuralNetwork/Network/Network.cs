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
		public Network(int inputSize, int hiddenSize, int outputSize)//, double? learnRate_output = null, double? learnRate_hidden = null, double? momentum = null)
		{
			//LearnRate_Output = learnRate_output ?? 0.5;
			//LearnRate_Hidden = learnRate_hidden ?? 0.7;
			//Momentum = momentum ?? 0.0043;
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
		//public void Train(List<DataSet> dataSets, int numEpochs)
		//{
		//	for (var i = 0; i < numEpochs; i++)
		//	{
		//		foreach (var dataSet in dataSets)
		//		{
		//			ForwardPropagate(dataSet.Values);
		//			BackPropagate(dataSet.Targets);
		//		}
		//	}
		//}

		public void Train(List<DataSet> dataSets, double minimumError)
		{
			var error = 1.0;
			var numEpochs = 0;

			while (error > minimumError && numEpochs < int.MaxValue)
			{
				var errors = new List<double>();

				if (200 < numEpochs && numEpochs < 801 && 6000 < numEpochs && numEpochs < 7001 )
				{
					LearnRate_Hidden = 0.4;
					LearnRate_Output = 0.2;
					Momentum = 0.2;
				}
				else if (800 < numEpochs && numEpochs < 1301 && 7000 < numEpochs && numEpochs < 8001)
				{
					LearnRate_Hidden = 0.2;
					LearnRate_Output = 0.15;
					Momentum = 0.05;
				}
				else if (1300 < numEpochs && numEpochs < 2001 && 8000 < numEpochs && numEpochs < 9001)
				{
					LearnRate_Hidden = 0.1;
					LearnRate_Output = 0.05;
					Momentum = 0.025;
				}
				else if (2000 < numEpochs && numEpochs < 5001 && 9000 < numEpochs && numEpochs < 10001)
				{
					LearnRate_Hidden = 0.05;
					LearnRate_Output = 0.025;
					Momentum = 0.01;
				}

				else if (5000 < numEpochs && numEpochs < 6001 && 10000 < numEpochs)
				{
					LearnRate_Hidden = 0.025;
					LearnRate_Output = 0.01;
					Momentum = 0.005;
				}
				else
				{
					LearnRate_Hidden = 0.7;
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
				//Console.WriteLine(LearnRate_Hidden);
				//Console.WriteLine(LearnRate_Output);
				//Console.WriteLine(Momentum);

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
			//return OutputLayer.Sum(a => Math.Abs(a.CalculateError(targets[i++])));
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