package ai2;

import java.util.ArrayList;

public class Node {
	// This class contains all a nodes information alongside some functions to retrieve or update data
	private ArrayList<Double> Wheight_values = new ArrayList<Double>();
	private double delta;
	private double weight_sum;
	private double output;

	public Node(int Input_number){
		double max_weight = (4 / (double) Input_number);
		double result;
		//Input_number + 1 to account for offset
		for (int i = 0; i < Input_number + 1; i++) {
			result = 0.1 + (Math.random() * (max_weight - 0.1));
			result += -(max_weight/2);
			Wheight_values.add(result);
		
		}
		delta = 0;
		weight_sum = 0;	
	}

	//Returns a specific weight value
	public double get(int location) {
		return Wheight_values.get(location);
	}
	
	//Changes the weights
	public void updateWeights(double learning_factor,ArrayList<Double> values) {
		double new_weight = 0;
		double old_weight = 0;
		for (int weight = 1; weight < Wheight_values.size();weight++) {
			old_weight = Wheight_values.get(weight);
			new_weight = old_weight + (learning_factor * delta * values.get(weight - 1)) ;
			Wheight_values.set(weight, new_weight);
			
		}
		old_weight = Wheight_values.get(0);
		new_weight = old_weight + (learning_factor * delta) ;
		Wheight_values.set(0, new_weight);
	}

	// Some general function to access the data
	public int size() {
		return Wheight_values.size();
	}	
	public double getOutput() {	
		return output;
	}
	public void setOutput(double i) {	
		output = i;
	}
	public double getWeightSum() {	
		return weight_sum;	
	}
	public void change_weight_sum(double i) {	
		weight_sum = i;
	}
	public double get_delta_value() {	
		return delta;
	}
	public void change_special_value(double i) {	
		delta = i;
	}
}
