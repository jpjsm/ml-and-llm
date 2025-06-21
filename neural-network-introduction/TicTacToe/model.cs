using NeuralNet;

namespace TicTacToe;

public class Model
{
    SimpleNeuralNetwork net;
    internal Model(SimpleNeuralNetwork network)
    {
        net = network;
    }

    public int BestSquare(TicTacToeGame game)
    {
        int bestSquare = 0;
        double bestScore = double.NegativeInfinity;

        foreach (int square in game.ValidSquares())
        {
            double[] boardAfter = game.GetBoardAsDouble(game.IsXTurn);
            boardAfter[square] = 1;
            net.PushInputValues(boardAfter);
            double score = net.GetOutput()[0];

            if (score > bestScore)
            {
                bestScore = score;
                bestSquare = square;
            }
        }
        return bestSquare;
    }
}
