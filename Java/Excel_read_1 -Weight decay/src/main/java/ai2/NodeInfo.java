package ai2;

import java.util.ArrayList;

public class NodeInfo {

	private static ArrayList<ArrayList<Node>> Wheight_values = new ArrayList<ArrayList<Node>>();
	// This declares the neural network to the users specifications including the nodes on each layer
	public NodeInfo(String[] Nodes,int nodeInputCount){
		ArrayList<Node> indervidual_node;
		for (int i = 0; i < Nodes.length;i++) {
			int layerNodecount = Integer.parseInt(Nodes[i]);
			if (layerNodecount <= 0) {
				System.err.println("Layer " + i + "outside acceptable range");
			}
			
			// This piece of code makes the program set the input to the previous layers node quality
			if (i != 0) {
				nodeInputCount = Integer.parseInt(Nodes[i - 1]);
			}

			indervidual_node = new ArrayList<Node>();
			for (int i2 = 0; i2 < layerNodecount; i2++) {
				Node value = new Node(nodeInputCount);
				indervidual_node.add(value);	
			}
			Wheight_values.add(indervidual_node);
		
		}
		Node Output_node_values = new Node(Integer.parseInt(Nodes[Nodes.length-1]));
		indervidual_node = new ArrayList<Node>();
		indervidual_node.add(Output_node_values);
		Wheight_values.add(indervidual_node);
	}

	// This function returns a arraylist containing all the outputs of a layer
	public static ArrayList<Double> getLayerOutputs(int layer){
		ArrayList<Double> values = new ArrayList<Double>();
		for (int i = 0; i < getNodeCount(layer);i++) {
			values.add(getNode(i,layer).getWeightSum());
		}
		return values;
	}

	// Returns the amount of nodes in a layer
	public static int getNodeCount(int layer) {
		return(Wheight_values.get(layer).size());
	}
	
	public static void updateNodeDelta (int node,int layer,double Sigmoid_FO,double correct_value,boolean finalNode,double epoc,double learning_rate) {
		double weight_number = getNode(node,layer).size();
	
		// Weight decay can be found here <---------------------------------------------------------------------------------------------------------------------------
		double omega = 0;
		for (int i = 0; i < weight_number;i++) {
			omega += getNode(node,layer).get(i) * getNode(node,layer).get(i);
		
		}	
		omega = omega*(1/(2*weight_number));
		double upsilon = 0.1/((epoc+1)*learning_rate);
		upsilon = Math.max(upsilon, 0.001);
		upsilon = Math.min(upsilon, 0.1);
		
		if (finalNode) {		
			double sum = ((correct_value - getNode(node,layer).getOutput())+(omega*upsilon)) * Sigmoid_FO;
			getNode(node,layer).change_special_value(sum);
			
		} else {
			double sum = 0;
			for (int i = 0; i < getNodeCount(layer + 1);i++) {
				sum += getNode(i,layer + 1).get(node + 1) * getNode(i,layer + 1).get_delta_value();
			}
			getNode(node,layer).change_special_value((sum) * Sigmoid_FO);	
		}
	}
	
	
	

	//Updates the weightsum for a specific node
	public static void updateNodeWeightSum(int node,int layer,ArrayList<Double> values) {

		double Weight_sum = getNode(node,layer).get(0);
		for (int i = 0; i < getNode(node,layer).size() - 1;i++) {
			Weight_sum += getNode(node,layer).get(i + 1) * values.get(i);
			
		}	
		getNode(node,layer).change_weight_sum(Weight_sum);
	}

	//Returns a specific node if direct access is needed
	public static Node getNode(int node,int layer) {	
		return(Wheight_values.get(layer).get(node));
	}

}
