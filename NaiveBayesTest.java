// NaiveBaseTest.java
// 2011.07.26 Hiroki Tanioka

import java.io.*;
import java.util.*;

/**
 * Test Class for NaiveBayes Classifier
 */
public class NaiveBayesTest {
	
	static private int DEF_TERM_NUM = 3000;
	static private int DEF_LABEL_NUM = 2;
	
	static public boolean logging = false;
	
	static NaiveBayes bayse = new NaiveBayes();
	
	static Map<String, Integer> cateMap = new HashMap<String, Integer>();
	static List<String> cateList = new ArrayList<String>();
	static Map<String, Integer> termMap = new HashMap<String, Integer>();
	static List<String> termList = new ArrayList<String>();
	
	public static void main(String args[]) {
		
		try {
			InputStreamReader isr = new InputStreamReader(System.in);
			BufferedReader r = new BufferedReader(isr);
			while (true) {
				System.out.println("");
				System.out.print("> ");
				String s = r.readLine();
				if (s == null || "".equals(s) || ".".equals(s)) {
					break;
				}
				String[] commands = {"--help", "/?", "train ", "choice ", "output", };
				int index = 0;
				String line = "";
				for (index = 0; index < commands.length; index++) {
					String command = commands[index];
					int pos = s.indexOf(command);
					if (pos >= 0) {
						line = s.substring(pos + command.length());
						break;
					}
				}
				switch (index) {
					case 2: 
					{
						//System.out.println(line);
						if (train(line) == false) {
							break;
						}
					}
					break;
					case 3: 
					{
						//System.out.println(line);
						if (choice(line) == false) {
							break;
						}
					}
					break;
					case 4: 
					{
						if (output(line) == false) {
							break;
						}
					}
					break;
					case 0: 
					case 1: 
					default: 
					{
						System.out.print(
"Usage: java NaiveBayesTest\n" + 
"           (to execute this Test tool) + \n" + 
"\n" + 
"where online commands include:\n" + 
"    train         is to train keywords for a category\n" + 
"                  <the head term can be interpreted as category.>\n" + 
"                  <the following terms can be interpreted as keywords.>\n" + 
"                  e.g.) > train enjoy leisure summer_vacation new_year_vacation\n" + 
"                        trained!\n" + 
"\n" + 
"    choice        is to choose a category for the given keywords\n" + 
"                  <all terms can be interpreted as keywords.>\n" + 
"                  e.g.) > choice leisure summer_vacation new_year_vacation\n" + 
"                        category = enjoy (0)\n" + 
"\n" + 
"    output        is to show keyword probabilities for each category.\n" + 
"                  <the following terms can be interpreted as categories.>\n" + 
"                  <If following term is none, all categories are shown.>\n" + 
"                  e.g.) > output enjoy\n" + 
"                        category, enjoy,\n" + 
"                        total, 3,\n" + 
"                        leisure, 73.1(1.0),\n" + 
"                        new_year_vacation, 73.1(1.0),\n" + 
"                        summer_vacation, 73.1(1.0),\n" + 
						"");
					}
					break;
				}
			}
			r.close();
			return;
		}
		catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}
	
	private static boolean output(String line) {
		if (cateMap.size() <= 0 || cateList.size() <= 0 || 
			termMap.size() <= 0 || termList.size() <= 0) {
			System.out.println(cateMap.size() + ", " + 
								cateList.size() + ", " + 
								termMap.size() + ", " + 
								termList.size());
			return false;
		}
		
		Map<String, Integer> termFreq = new HashMap<String, Integer>();
		
		String[] cates = line.split(" ");
		if (cates == null) {
			System.out.println("Failed to split.");
			return false;
		}
		
		Set<String> category_set = new TreeSet<String>();
		for (int i = 0; i < cates.length; i++) {
			String term = cates[i].trim();
			if (term == null || "".equals(term)) {
				continue;
			}
			category_set.add(term);
		}
		//System.out.println("category_set = " + category_set.size());
		
		List<Integer> categories = new ArrayList<Integer>();
		List<float[]> probs = new ArrayList<float[]>();
		List<float[]> terms = new ArrayList<float[]>();
		bayse.output(categories, probs, terms);
		
		// categories
		System.out.print("category, ");
		for (int i = 0; i < categories.size(); i++) {
			String category = cateList.get(i);
			if (category_set.size() > 0 && !category_set.contains(category)) {
				continue;
			}
			System.out.print(category + ", ");
		}
		System.out.println("");
		System.out.print("total, ");
		for (int i = 0; i < categories.size(); i++) {
			String category = cateList.get(i);
			if (category_set.size() > 0 && !category_set.contains(category)) {
				continue;
			}
			int cate_freq = categories.get(i).intValue();
			System.out.print(cate_freq + ", ");
		}
		System.out.println("");
		
		// term's probabilities
		for (int i = 0; i < terms.size(); i++) {
			String term = termList.get(i);
			System.out.print(term + ", ");
			float[] term_probs = probs.get(i);
			float[] term_freqs = terms.get(i);
			for (int j = 0; j < term_probs.length; j++) {
				String category = cateList.get(j);
				if (category_set.size() > 0 && 
					!category_set.contains(category)) {
					continue;
				}
				System.out.print(term_probs[j] + 
							"(" + term_freqs[j] + ")" + ", ");
			}
			System.out.println("");
		}
		
		return true;
	}
	
	private static boolean choice(String line) {
		if (cateMap.size() <= 0 || cateList.size() <= 0 || 
			termMap.size() <= 0 || termList.size() <= 0) {
			return false;
		}
		
		Map<String, Integer> termFreq = new HashMap<String, Integer>();
		
		String[] terms = line.split(" ");
		if (terms.length <= 0) {
			System.err.println("Failed to split.");
			return false;
		}
		
		for (int i = 0; i < terms.length; i++) {
			String term = terms[i].trim();
			if (term == null || "".equals(term)) {
				continue;
			}
			Integer _freq = termFreq.get(term);
			if (_freq == null) {
				_freq = new Integer(0);
				termFreq.put(term, _freq);
			}
			int freq = _freq.intValue();
			termFreq.put(term, new Integer(freq + 1));
		}
		
		if (logging) {
			Set<String> termSet = termFreq.keySet();
			for (Iterator<String> i = termSet.iterator(); i.hasNext();) {
				String key = i.next();
				Integer freq = termFreq.get(key);
				System.out.println(key + ": " + freq);
			}
		}
		
		int[] term_id = new int[termFreq.size()];
		float[] term_fq = new float[termFreq.size()];
		
		int index = 0;
		Set<String> termSet = termFreq.keySet();
		for (Iterator<String> i = termSet.iterator(); i.hasNext();) {
			String key = i.next();
			Integer freq = termFreq.get(key);
			
			Integer _id = termMap.get(key);
			if (_id == null) {
//				int id = termList.size();
				_id = new Integer(-1);
				//System.out.println(key + " " + _id.toString());
//				termMap.put(key, _id);
//				termList.add(key);
			}
			//System.out.println(key + " " + termMap.get(key));
			term_id[index] = _id.intValue();
			term_fq[index] = freq.intValue();
			index++;
		}
		
		int result_id = bayse.choice(term_id, term_fq);
		String category = cateList.get(result_id);
		System.out.println("category = " + category + " (" + result_id + ")");
		
		return true;
	
	}
	
	private static boolean train(String line) {
		final int repeat = 10;
		return train(line, repeat);
	}
	
	private static boolean train(String line, int repeat) {
		if (cateMap.size() <= 0 || cateList.size() <= 0 || 
			termMap.size() <= 0 || termList.size() <= 0) {
			repeat = 0;
		}
		
		Map<String, Integer> termFreq = new HashMap<String, Integer>();
		
		String[] terms = line.split(" ");
		if (terms.length <= 0) {
			System.out.println("Failed to split.");
			return false;
		}
		
		String category = terms[0].trim();
		Integer _cate_id = cateMap.get(category);
		if (_cate_id == null) {
			int cate_id = cateList.size();
			_cate_id = new Integer(cate_id);
			cateMap.put(category, cate_id);
			cateList.add(category);
		}
		
		for (int i = 1; i < terms.length; i++) {
			String term = terms[i].trim();
			if (term == null || "".equals(term)) {
				continue;
			}
			Integer _freq = termFreq.get(term);
			if (_freq == null) {
				_freq = new Integer(0);
				termFreq.put(term, _freq);
			}
			int freq = _freq.intValue();
			termFreq.put(term, new Integer(freq + 1));
		}
		
		if (logging) {
			System.out.println("category = " + _cate_id.intValue());
			Set<String> termSet = termFreq.keySet();
			for (Iterator<String> i = termSet.iterator(); i.hasNext();) {
				String key = i.next();
				Integer freq = termFreq.get(key);
				System.out.println(key + ": " + freq);
			}
		}
		
		int[] term_id = new int[termFreq.size()];
		float[] term_fq = new float[termFreq.size()];
		
		int index = 0;
		Set<String> termSet = termFreq.keySet();
		for (Iterator<String> i = termSet.iterator(); i.hasNext();) {
			String key = i.next();
			Integer freq = termFreq.get(key);
			
			Integer _id = termMap.get(key);
			if (_id == null) {
				int id = termList.size();
				_id = new Integer(id);
				//System.out.println(key + " " + _id.toString());
				termMap.put(key, _id);
				termList.add(key);
			}
			//System.out.println(key + " " + termMap.get(key));
			term_id[index] = _id.intValue();
			term_fq[index] = freq.intValue();
			index++;
		}
		
		//System.out.println("cate_id = " + _cate_id);
		boolean ret = bayse.train(_cate_id.intValue(), term_id, term_fq);
		if (ret == false) {
			System.out.println("Failed to train.");
		}
		if (repeat > 0) {
			// conditional train
			for (int i = 1; i < repeat; i++) {
				int result_id = bayse.choice(term_id, term_fq);
				if (result_id == _cate_id.intValue()) {
					break;
				}
				ret = bayse.train(_cate_id.intValue(), term_id, term_fq);
				if (ret == false) {
					System.out.println("Failed to train.");
				}
			}
		}
		
		int result_id = bayse.choice(term_id, term_fq);
		if (result_id == _cate_id.intValue()) {
			System.out.println("trained!");
		}
		else {
			System.out.println("insufficiency...");
		}
		
		return true;
	}
	
}
