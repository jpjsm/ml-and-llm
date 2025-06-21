// See https://aka.ms/new-console-template for more information
using NeuralNet;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System;
using System.Linq;

namespace HandWrittenNumberRecognition
{
    public record Mnist_data_row
    {
        private static readonly HashSet<string> valid_labels = new HashSet<string>(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] );
        private double[] standardized_pixel_values;

        public static  IReadOnlyCollection<string> Valid_Labels
        {
            get { return valid_labels;}
        }

        public static int Max_pixel_value => 255;

        public static int Min_pixel_value => 0;
        public string Expected_label {get; init;}
        public int[] Pixel_values {get; init;}
        public double[] Standardized_Pixel_Values => standardized_pixel_values;
        public Mnist_data_row(string expected_label, IList<string> values)
        {
            if (!valid_labels.Contains(expected_label)) throw new ArgumentException($"{nameof(expected_label)}: '{expected_label}' not a valid label.");
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (values.Count != Mnist_dataset.Image_Pixels) throw new ArgumentException($"Not the expected number of elements in {nameof(values)}: '{values.Count}' != '{Mnist_dataset.Image_Pixels}'");
            int[] pixel_values = values.Select(v => Int32.Parse(v)).ToArray();
            if (pixel_values.Any(v => v < Min_pixel_value || v > Max_pixel_value))  throw new ArgumentException($"One or more of the pixel values is outside the valid value range [0..255]");
            this.Expected_label = expected_label;
            this.Pixel_values = pixel_values;
            double mpv = Convert.ToDouble(Max_pixel_value);
            this.standardized_pixel_values = pixel_values.Select(p => Convert.ToDouble(p)/mpv).ToArray();
            //string standardized_pixel_values_str = string.Join(", ", standardized_pixel_values.Select(p => $"{p:f6}"));
            //Console.WriteLine($"standardized_pixel_values: [{standardized_pixel_values_str}]");
        }
    }
}

