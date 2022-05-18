using System;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;

namespace Sentiment_Client
{
    class Program
    {
        private static readonly string _url = "http://atmosera.northcentralus.azurecontainer.io:8008/analyze";

        static async Task Main(string[] args)
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

            // Pass the text to the Web service
            var client = new HttpClient();
            var url = _url + $"?text={text}";
            var response = await client.GetAsync(url);
            var score = await response.Content.ReadAsStringAsync();
            Console.WriteLine(score);
        }
    }
}
