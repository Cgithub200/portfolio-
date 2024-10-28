package ai2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.Scanner;


public class MainProgramWD {

    private static final double LEARNING_RATE = 0.1;
    private static double weightSum;
    private static double sigmoid;
    private static double sigmoidDerivative;
    private static double correctValue;
    private static NodeInfo weights;
    
    public static void main(String[] args) {
        // Collect necessary inputs for the program
        Scanner scanner = new Scanner(System.in);
        
        // Load data from Excel file and prints exception
        String filePath = "./Data/DataSet.xlsx";
        ExcelJavaUtility memory = new ExcelJavaUtility(filePath, "Training");
		

        // Get the number of nodes from user input
        System.out.println("Enter the number of nodes in each layer (comma separated):");
        String[] nodeAmounts = scanner.nextLine().split(",");
        int layerCount = nodeAmounts.length;
        int inputCount = memory.getColumnCount() - 1;

        System.out.println("Input count: " + inputCount);

        // Initialize weights
        weights = new NodeInfo(nodeAmounts, inputCount);

        double previousValidationError = Double.MAX_VALUE;
        int maxEpochs = 10000;

        // Main training loop
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            runTraining(memory, layerCount, epoch);

            // Validate every 500 epochs to prevent overtraining
            if (epoch % 500 == 0) {
                memory.changeSheet("Validation");
                double validationError = calculateMeanAbsoluteError(memory, layerCount);

                System.out.println("Validation error: " + validationError);

                if (previousValidationError < validationError) {
                    System.out.println("Overtraining detected, stopping training.");
                    break;
                }

                previousValidationError = validationError;
                memory.changeSheet("Training");
            }
        }

        // Perform testing on the final model
        memory.changeSheet("Testing");
        double testingError = calculateMeanAbsoluteError(memory, layerCount);
        System.out.println("Testing error: " + testingError);
    }
	
    private static void runTraining(ExcelJavaUtility memory, int layerCount, int epoch) {
        for (int i = 0; i < memory.getRowCount() - 1; i++) {
            // Forward pass
            correctValue = memory.getRowCellData(i).get(memory.getRowCellData(i).size() - 1);
            forwardPass(memory, layerCount, i);

            // Backward pass and weight updates
            backwardPass(layerCount, epoch);
            updateWeights(memory, layerCount, i);
        }
    }

    private static void forwardPass(ExcelJavaUtility memory, int layerCount, int rowIndex) {
        // Pass through the first hidden node layer
        for (int node = 0; node < weights.getNodeCount(0); node++) {
            weights.updateNodeWeightSum(node, 0, memory.getRowCellData(rowIndex));
            sigmoid = sigmoidActivation(weights.getNode(node, 0).getWeightSum());
            weights.getNode(node, 0).setOutput(sigmoid);
        }

        // Pass through subsequent layers
        for (int layer = 1; layer <= layerCount; layer++) {
            for (int node = 0; node < weights.getNodeCount(layer); node++) {
                weights.updateNodeWeightSum(node, layer, weights.getLayerOutputs(layer - 1));
                sigmoid = sigmoidActivation(weights.getNode(node, layer).getWeightSum());
                weights.getNode(node, layer).setOutput(sigmoid);
            }
        }
    }

    private static void backwardPass(int layerCount, int epoch) {
        sigmoid = weights.getNode(0, layerCount).getOutput();
        sigmoidDerivative = sigmoidDerivative(sigmoid);
        weights.updateNodeDelta(0, layerCount, sigmoidDerivative, correctValue, true, epoch, LEARNING_RATE);

        for (int layer = layerCount - 1; layer >= 0; layer--) {
            for (int node = 0; node < weights.getNodeCount(layer); node++) {
                sigmoid = weights.getNode(node, layer).getOutput();
                sigmoidDerivative = sigmoidDerivative(sigmoid);
                weights.updateNodeDelta(node, layer, sigmoidDerivative, 0, false, epoch, LEARNING_RATE);
            }
        }
    }

    private static void updateWeights(ExcelJavaUtility memory, int layerCount, int rowIndex) {
        ArrayList<Double> rowData = new ArrayList<>(memory.getRowCellData(rowIndex));
        rowData.remove(rowData.size() - 1);  // Remove the expected output value

        // Update weights for the first layer
        for (int node = 0; node < weights.getNodeCount(0); node++) {
            weights.getNode(node, 0).updateWeights(LEARNING_RATE, rowData);
        }

        // Update weights for subsequent layers
        for (int layer = 1; layer <= layerCount; layer++) {
            for (int node = 0; node < weights.getNodeCount(layer); node++) {
                weights.getNode(node, layer).updateWeights(LEARNING_RATE, weights.getLayerOutputs(layer - 1));
            }
        }
    }

    private static double calculateMeanAbsoluteError(ExcelJavaUtility memory, int layerCount) {
        ArrayList<Double> outputErrors = new ArrayList<>();

        for (int i = 0; i < memory.getRowCount() - 1; i++) {
            forwardPass(memory, layerCount, i);

            double output = weights.getNode(0, layerCount).getOutput();
            correctValue = memory.getRowCellData(i).get(memory.getRowCellData(i).size() - 1);
            outputErrors.add(Math.abs(correctValue - output));
        }

        return outputErrors.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    // Activation function and its derivative
    private static double sigmoidActivation(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    private static double sigmoidDerivative(double sigmoidValue) {
        return sigmoidValue * (1 - sigmoidValue);
    }
}
