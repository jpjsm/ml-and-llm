// See https://aka.ms/new-console-template for more information
using NeuralNet;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System;
using System.Linq;

namespace HandWrittenNumberRecognition
{
    public static class Mnist_dataset
    {
        private static Random rnd = new Random();
        private static Dictionary<Dataset_types, Mnist_data_row[]> datasets = new Dictionary<Dataset_types, Mnist_data_row[]>(){
            {Dataset_types.Train, new Mnist_data_row[0]},
            {Dataset_types.Test, new Mnist_data_row[0]}
        };
        private static Dictionary<string, List<double>> expected_results = new Dictionary<string, List<double>> (){
            {"0", [0.982, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018]},
            {"1", [0.018, 0.982, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018]},
            {"2", [0.018, 0.018, 0.982, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018]},
            {"3", [0.018, 0.018, 0.018, 0.982, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018]},
            {"4", [0.018, 0.018, 0.018, 0.018, 0.982, 0.018, 0.018, 0.018, 0.018, 0.018]},
            {"5", [0.018, 0.018, 0.018, 0.018, 0.018, 0.982, 0.018, 0.018, 0.018, 0.018]},
            {"6", [0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.982, 0.018, 0.018, 0.018]},
            {"7", [0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.982, 0.018, 0.018]},
            {"8", [0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.982, 0.018]},
            {"9", [0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.982]},
        };
        public static readonly (int rows, int columns) Image_Size = (28, 28);
        public static readonly  int Image_Pixels = Image_Size.rows * Image_Size.columns; // Image_Size.rows * Image_Size.columns;


        public static ReadOnlyDictionary<string, List<double>> Expected_Results
        {
            get { return new ReadOnlyDictionary<string, List<double>>(expected_results); }
        }
        
        public static Mnist_data_row[] Train_dataset 
        {
            get { return Dataset_sample(Dataset_types.Train, datasets[Dataset_types.Train].Length); } 
        }

        public static Mnist_data_row[] Test_dataset
        {
            get { return Dataset_sample(Dataset_types.Test, datasets[Dataset_types.Test].Length); } 
        }

        public static void Load_Mnist_dataset(string train_dataset_filename, string test_dataset_filename)
        {
            if (String.IsNullOrWhiteSpace(train_dataset_filename)) throw new ArgumentNullException(nameof(train_dataset_filename));
            if (!File.Exists(train_dataset_filename)) throw new FileNotFoundException(nameof(train_dataset_filename));
            if (String.IsNullOrWhiteSpace(test_dataset_filename)) throw new ArgumentNullException(nameof(test_dataset_filename));
            if (!File.Exists(test_dataset_filename)) throw new FileNotFoundException(nameof(test_dataset_filename));

            List<Mnist_data_row> train_dataset = [];
            foreach (string line in File.ReadLines(train_dataset_filename))
            {
                string[] elements = line.Split(",");
                Mnist_data_row dr = new Mnist_data_row(elements[0], elements[1..]);
                train_dataset.Add(dr);
            }

            List<Mnist_data_row> test_dataset = [];
            foreach (string line in File.ReadLines(test_dataset_filename))
            {
                string[] elements = line.Split(",");
                Mnist_data_row dr = new Mnist_data_row(elements[0], elements[1..]);
                test_dataset.Add(dr);
            }

            datasets[Dataset_types.Train] = train_dataset.ToArray();
            datasets[Dataset_types.Test] = test_dataset.ToArray();
        }

        public static Mnist_data_row[] Dataset_sample(Dataset_types dtype, int rows_number)
        {
            if (rows_number < 0) throw new ArgumentException("number of rows must be a positive number");
            
            if (rows_number >= datasets[dtype].Length)
            {
                Mnist_data_row[] result = new Mnist_data_row[datasets[dtype].Length];
                Array.Copy(datasets[dtype], result, datasets[dtype].Length);
                return result;   
            }

            List<int> bag = [];
            for (int i = 0; i < datasets[dtype].Length; i++)
            {
                bag.Add(i);
            }

            List<Mnist_data_row> sample_set = [];
            for (int i = 0; i < rows_number; i++)
            {
                int r = rnd.Next(0, bag.Count);
                sample_set.Add(datasets[dtype][bag[r]]);
                bag.Remove(r);
            }

            return sample_set.ToArray();
        }
    }

}

