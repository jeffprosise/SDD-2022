using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Diagnostics.Contracts;

namespace MLN_SentimentClient
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\Sentiment.zip";

        static void Main(string[] args)
        {
            string text;

            // Get the text to analyze
            if (args.Length > 0)
            {
                text = args[0];
            }
            else
            {
                Console.Write("Text to analyze: ");
                text = Console.ReadLine();
            }

            // Load the model
            var context = new MLContext(seed: 0);
            var model = context.Model.Load(_path, out DataViewSchema schema);

            // Create a prediction engine and use it to score the input text
            var input = new Input { SentimentText = text };
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);
            var prediction = predictor.Predict(input);
            Console.WriteLine(prediction.Probability);
        }
    }

    public class Input
    {
        public string SentimentText;
    }

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}
