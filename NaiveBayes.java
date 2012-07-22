// NaiveBayes.java
// 2005.07.20 Hiroki Tanioka	first implementation.
// 2005.07.23 Hiroki Tanioka	refine.
// 2005.12.14 Hiroki Tanioka	refine.
// 2011.07.23 Hiroki Tanioka	reform.

import java.io.*;
import java.util.*;
import java.math.BigDecimal;

/**
 * NaiveBayes Classifier Class
 */
public class NaiveBayes {
	
	static private int DEF_TERM_NUM = 3000;
	static private int DEF_LABEL_NUM = 1;
	
	private Object[] termMap = null;
	private float[] totalMap = null;
	
	static public boolean logging = false;
	
	// constructor
	public NaiveBayes() {
		termMap = new Object[DEF_TERM_NUM];
		totalMap = new float[DEF_LABEL_NUM];
	}
	
	public NaiveBayes(int size) {
		termMap = new Object[DEF_TERM_NUM];
		totalMap = new float[size];
	}
	
	// load model
	public boolean load(String fine_name) {
		try {
			FileInputStream fis = new FileInputStream(fine_name);
			InputStreamReader isr = new InputStreamReader(fis);
			BufferedReader br = new BufferedReader(isr);
			
			String line = br.readLine();
			if(line == null) {
				return false;
			}
			StringTokenizer st = new StringTokenizer(line, ",");
			int size = st.countTokens();
			if (size <= 0) {
				return false;
			}
			totalMap = new float[size];
			for (int i = 0; st.hasMoreTokens(); i++) {
				String tmp = st.nextToken();
				totalMap[i] = Float.parseFloat(tmp);
			}
			
			line = br.readLine();
			if (line == null) {
				return false;
			}
			size = Integer.parseInt(line);
			termMap = new Object[size];
			for (int i = 0; i < size; i++) {
				line = br.readLine();
				if(line == null) {
					break;
				}
				st = new StringTokenizer(line, ",:");
				int count = st.countTokens();
				if (count <= 1 || st.hasMoreTokens() == false) {
					continue;
				}
				String tmp = st.nextToken();	// skipping
				float[] labelMap = new float[count - 1];
				for (int j = 0; st.hasMoreTokens(); j++) {
					tmp = st.nextToken();
					labelMap[j] = Float.parseFloat(tmp);
				}
				termMap[i] = labelMap;
			}
			
			br.close();
		}
		catch(Exception e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	// output terms
	public boolean output(List<Integer> categories, 
							List<float[]> probs, List<float[]> terms) {
		if (totalMap == null || termMap == null) {
			return false;
		}
		
		try {
			int total_length = totalMap.length;
			if (logging) {
				System.out.print("category: ");
				for (int i = 0; i < total_length; i++) {
					System.out.print(i + ", ");
				}
				System.out.println("");
				System.out.print("total: ");
				for (int i = 0; i < total_length; i++) {
					System.out.print(totalMap[i] + ", ");
				}
				System.out.println("");
			}
			for (int i = 0; i < total_length; i++) {
				categories.add(new Integer((int)totalMap[i]));
			}
			
			int term_length = termMap.length;
			//System.out.println("term_length = " + term_length);
			
			for (int i = 0; i < term_length; i++) {
				final int term = i;
				float[] term_probs = new float[total_length];
				float[] term_freqs = new float[total_length];
				
				double ham = 0;
				double spam = 0;
				
				if (logging) {
					System.out.print(term + ": ");
				}
				int total_freq = 0;
				double likelihood = 0;
				for (int j = 0; j < total_length; j++) {
					int label = j;
					//System.out.print(labelMap[j] + ",");
					final double numer = numerate(label, term);
					if (numer < 0) {
						likelihood = 0;
					}
					else {
						final double denom = denominate(term);
						if (denom <= 0) {
							continue;
						}
						likelihood = (float)(numer / denom);
					}
					
					// belief (0.5 ~ 1.0)
					double term_count = 0;
					float[] labelMap = (float[])termMap[term];
					if (labelMap != null && labelMap.length > label) {
						term_count = (double)labelMap[label];
					}
					likelihood *= (1 / (1 + Math.exp(-1 * term_count)));
					
					likelihood *= 100;
					BigDecimal lh = new BigDecimal(likelihood);
					double _lh = lh.setScale(1, BigDecimal.ROUND_DOWN).doubleValue();
					if (logging) {
						System.out.print(_lh + "%, ");
					}
					term_probs[j] = (float)_lh;
					
					double freq = 0;
					//float[] labelMap = (float[])termMap[term];
					if (labelMap != null && labelMap.length > label) {
						freq = labelMap[label];
					}
					total_freq += freq;
					term_freqs[j] = (float)freq;
				}
				if (logging) {
					System.out.println("");
				}
				if (total_freq > 0) {
					probs.add(term_probs);
					terms.add(term_freqs);
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	// save model
	public boolean save(String file_name) {
		if (totalMap == null || termMap == null) {
			return false;
		}
		
		try {
			FileOutputStream fos = new FileOutputStream(file_name);
			OutputStreamWriter osr = new OutputStreamWriter(fos);
			BufferedWriter bw = new BufferedWriter(osr);
			
			int total_length = totalMap.length;
			for (int i = 0; i < total_length; i++) {
				bw.write(totalMap[i] + ",");
			}
			bw.newLine();
			
			int term_length = termMap.length;
			bw.write("" + term_length);
			bw.newLine();
			
			for (int i = 0; i < term_length; i++) {
				bw.write(i + ":");
				float[] labelMap = (float[])termMap[i];
				if (labelMap == null) {
					bw.newLine();
					continue;
				}
				int label_length = labelMap.length;
				for (int j = 0; j < label_length; j++) {
					bw.write(labelMap[j] + ",");
				}
				bw.newLine();
			}
			
			bw.close();
		}
		catch(Exception e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	// train
	public boolean train(int label, int[] terms, float[] freqs) {
//		System.out.println(label + " **********");
		final int terms_size = terms.length;
		int term_size = termMap.length;
		int label_size = totalMap.length;
		if (label >= label_size) {
			// ここで拡張
			int _size = label + 1;
			float[] _totalMap = new float[_size];
			System.arraycopy(totalMap, 0, _totalMap, 0, label_size);
			totalMap = _totalMap;
//			System.out.print(label_size);
			label_size = _size;
//			System.out.println("->" + label_size);
			//return false;
		}
		
		for (int i = 0; i < terms_size; i++) {
			final int term = terms[i];
			if (term >= term_size) {
				// ここで拡張
				int _size = term + term_size;
				Object[] _termMap = new Object[_size];
				System.arraycopy(termMap, 0, _termMap, 0, term_size);
				termMap = _termMap;
				term_size = _size;
				//continue;
			}
			final float freq = freqs[i];
			totalMap[label] += freq;
			
			// labelMap from termMap
			float[] labelMap = (float[])termMap[term];
			if (labelMap == null) {
				labelMap = new float[label_size];
				// Arrays.fill(labelMap, 0L);
				termMap[term] = labelMap;
			}
//			System.out.println("term = " + term + ", freq = " + freq);
			final int label_length = labelMap.length;
			if (label_length <= label) {
				// ここで拡張
				float[] _labelMap = new float[label + 1];
				System.arraycopy(labelMap, 0, _labelMap, 0, label_length);
				labelMap = _labelMap;
//				System.out.print(label_length);
//				System.out.println("->" + (label + label_length));
				termMap[term] = labelMap;
			}
			labelMap[label] += freq;
			
//			for (int j = 0; j < labelMap.length; j++) {
//				System.out.println(labelMap[j]);
//			}
//			System.out.println("--");
		}
		
//		for (int i = 0; i < totalMap.length; i++) {
//			System.out.println(totalMap[i]);
//		}
//		System.out.println("--");
		
		return true;
	}
	
	public double[] predict(int[] terms, float[] freqs) {
		final int term_size = termMap.length;
		final int terms_size = terms.length;
		final int label_size = totalMap.length;
		double[] oddses = new double[label_size];
		double total_odds = 0;
		double small_odds = 0;
		
		//System.out.println("terms_size = " + terms_size);
		//for (int i = 0; i < terms.length; i++) {
		//	System.out.print("" + terms[i] + ",");
		//}
		//System.out.println("");
		//for (int i = 0; i < freqs.length; i++) {
		//	System.out.print("" + freqs[i] + ",");
		//}
		//System.out.println("");
		
		for (int label = 0; label < label_size; label++) {
			if (logging) {
				System.out.println("label = " + label);
			}
			double odds = 0;
			double likelihood = 0;
			//double logx = 0;
			for (int i = 0; i < terms_size; i++) {
				final int term = terms[i];
				if (term < 0) {
					continue;
				}
				if (term >= term_size) {
					//System.err.println("Failed to short of term size.");
					continue;
				}
				final float freq = freqs[i];
				if (freq <= 0) {
					//System.out.println("freq = " + freq);
					continue;
				}
				double prob = 0;
				double numer = numerate(label, term);
				//System.out.println("numer = " + numer);
				if (numer <= 0) {
					//continue;
					prob = 0.5;
				}
				else {
					final double denom = denominate(term);
					//System.out.println("denom = " + denom);
					if (denom <= 0) {
						continue;
					}
					prob = numer / denom;
				}
				if (likelihood == 0) {
					likelihood = 1;
				}
				likelihood *= prob * freq;
				
				// belief (0.5 ~ 1.0)
				double term_count = 0;
				float[] labelMap = (float[])termMap[term];
				if (labelMap != null && labelMap.length > label) {
					term_count = (double)labelMap[label];
				}
				likelihood *= (1 / (1 + Math.exp(-1 * term_count)));
				//System.out.println("freq = " + freq + " likelihood *= " + likelihood);
			}
			//System.out.println("likelihood = " + likelihood);
			if (likelihood <= 0) {
				likelihood = small_odds;
			}
			odds = likelihood;
			oddses[label] = odds;// - small_odds;
			total_odds += oddses[label];
			if (logging) {
				System.out.println("odds = " + oddses[label]);
			}
		}
		
		//System.out.println("------------------------");
		//System.out.println("total = " + total_odds);
		for (int i = 0; i < oddses.length; i++) {
			double value = oddses[i];
			if (logging) {
				System.out.print("odds = " + value);
			}
			oddses[i] /= total_odds;
			if (logging) {
				System.out.println(", rate = " + oddses[i]);
			}
		}
		//System.out.println("------------------------");
		
		return oddses;
	}
	
	public int choice(int[] terms, float[] freqs) {
		double[] oddses = predict(terms, freqs);
		
		double max = Double.NEGATIVE_INFINITY;
		int best = -1;
		for (int label = 0; label < oddses.length; label++) {
			double likelihood = oddses[label];
			
			if (logging) {
				System.out.println(" likelihood = " + likelihood);
			}
			double odds = likelihood;
			if (max >= odds) {
				continue;
			}
			if (logging) {
				System.out.println(" max = " + odds);
			}
			max = odds;
			best = label;
		}
		
		if (best < 0) {
			System.err.println("Failed to wrong best label = " + best);
		}
		
		return best;
	}
	
	// probability numerator
	private double numerate(int label, int term) {
		//System.err.print("" + label + " " + term + " : ");
		final int term_size = termMap.length;
		// labelMap from termMap
		if (term_size <= term) {
			System.err.println("Failed to term length is too short. " + term);
			return -1;
		}
		float[] labelMap = (float[])termMap[term];
		if (labelMap == null) {
			//System.err.println("Failed to label length is too short.");
			return -1;
		}
		
		if (labelMap.length <= label) {
			//System.err.println("Failed to label length. " + labelMap.length);
			return 0;
		}
		double freq = labelMap[label];
		final double total = totalMap[label];
		if (total <= 0) {
			//System.err.println("Unknown Term!!");
			return 0;
		}
		if (freq <= 0) {
			// base-up
			freq = 0;
		}
		double p = freq / total;
		//System.err.println("" + freq + " / " + total + " = " + p);
		
		return p;
	}
	
	// probability denominator
	private double denominate(int term) {
		final int term_size = termMap.length;
		final int label_size = totalMap.length;
		//System.out.println("term = " + term);
		
		// labelMap from termMap
		if (term_size <= term) {
			System.err.println("Failed to term length is too short. " + term);
			return -1;
		}
		float[] labelMap = (float[])termMap[term];
		if (labelMap == null) {
			//System.err.println("Failed to label length is too short.");
			return -1;
		}
		
		// freqs from labelMap
		double denom = 0;
		final int label_length = labelMap.length;
		for (int i = 0; i < label_size; i++) {
			double freq = i < label_length ? labelMap[i] : 0;
			double total = totalMap[i];
			
			//System.out.println("freq = " + freq + ", total = " + total + " ");
			if (total <= 0) {
				continue;
			}
			double p = freq / total;
			denom += p;
		}
		//System.out.println("denom = " + denom);
		
		return denom;
	}
	
	public static void main(String args[]) {
		if (args.length <= 0) {
			System.out.println("usage: java Bayse " + 
								"[SEKI, HANA, ASE, ONDO, HIYAKE]");
			return;
		}
		
		int seki = Integer.parseInt(args[0]);
		int hana = Integer.parseInt(args[1]);
		int ase = Integer.parseInt(args[2]);
		int ondo = Integer.parseInt(args[3]);
		int hiyake = Integer.parseInt(args[4]);
		int high = Integer.parseInt(args[5]);
		
		// statement
		final int KENKO = 0;
		final int KAZE = 2;
		final int SUPER = 10;
		// phenomenon
		final int SEKI = 0;
		final int HANA = 1;
		final int ASE = 2;
		final int ONDO = 3;
		final int HIYAKE = 4;
		final int HIGH = 5;
		
		int byome;
		NaiveBayes bayse = new NaiveBayes();
		
		System.out.println("----------- train --------------");
		
		// train
		int[] shojos = 		{SEKI, 	HANA, 	ASE, 	ONDO, 	HIYAKE, 	HIGH	};
		float[] freqs_ken = {3, 	5, 		3, 		0, 		70, 	0, 	};
		float[] freqs_kaz = {5, 	10, 	3, 		10, 	5, 		0, 	};
		System.out.println("shojos = {SEKI, HANA, ASE, ONDO, HIYAKE, HIGH}");
		System.out.println("freqs_ken = {3, 5, 3, 0, 70, 0}");
		System.out.println("freqs_kaz = {5, 10, 3, 10, 5, 0}");
		// KENKO
		byome = KENKO;
		bayse.train(byome, shojos, freqs_kaz);
		bayse.train(byome, shojos, freqs_ken);
		bayse.train(byome, shojos, freqs_ken);
		bayse.train(byome, shojos, freqs_ken);
		// KAZE
		byome = KAZE;
		bayse.train(byome, shojos, freqs_ken);
		bayse.train(byome, shojos, freqs_kaz);
		bayse.train(byome, shojos, freqs_ken);
		bayse.train(byome, shojos, freqs_kaz);
		
		int[] shojos_x = {HIGH};
		float[] freqs_x = {1};
		byome = SUPER;
		bayse.train(byome, shojos_x, freqs_x);
		
		int[] shojos_y = {HIYAKE, HIGH};
		float[] freqs_y = {1, 1};
		byome = SUPER;
		bayse.train(byome, shojos_y, freqs_y);
		
		System.out.println("----------- predict --------------");
		
		// predict
		double[] byomes = null;
		float[] freqs = new float[shojos.length];
		freqs[0] = seki;
		freqs[1] = hana;
		freqs[2] = ase;
		freqs[3] = ondo;
		freqs[4] = hiyake;
		freqs[5] = high;
		byomes = bayse.predict(shojos, freqs);
		//for (int i = 0; i < byomes.length; i++) {
		//	System.out.println(i + " = " + byomes[i]);
		//}
		System.out.println("KENKO: " + byomes[KENKO] + "[%]");
		System.out.println("KAZE: " + byomes[KAZE] + "[%]");
		System.out.println("SUPER: " + byomes[SUPER] + "[%]");
		
		System.out.println("----------- choice --------------");
		
		int result = -1;
		//for (int i = 0; i < 100000; i++) {
			result = bayse.choice(shojos, freqs);
		//}
		System.out.println("KEKKA = " + result);
		
		System.out.println("----------- output --------------");
		List<Integer> categories = new ArrayList<Integer>();
		List<float[]> probs = new ArrayList<float[]>();
		List<float[]> terms = new ArrayList<float[]>();
		bayse.output(categories, probs, terms);
	}
}
