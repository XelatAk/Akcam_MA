using System;

namespace NeuralNetwork.Network
{
	public static class Sigmoid
	{
		public static double Output(double x)
		{
			// if, else if, else
			return x < -100.0 ? 0.0 : x > 100.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
		}

		public static double Derivative(double x)
		{
			return  x * (1 - x);
		}
	}
}