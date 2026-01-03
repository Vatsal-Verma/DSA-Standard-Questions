# infosys prep

### Combination Sum 
```

class Solution {
    public void track(int[] arr, int target, int index, List<Integer> lst, List<List<Integer>> result) {
        if(target == 0) {
            result.add(new ArrayList<>(lst));
            return;
        }

        if(target < 0) return ;
        for(int i=index; i<arr.length; i++ ) {  
            lst.add(arr[i]);

            track(arr, target - arr[i], i, lst, result);
            lst.remove(lst.size() - 1);
        }
    }

   
    public List<List<Integer>> combinationSum(int[] arr, int target) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> lst = new ArrayList<>();
        track(arr, target, 0, lst, result);
        return result;
    }
}

```

### Roman to Integer
```
class Solution {
    public int romanToInt(String s) {

        int result = 0;

        HashMap<Character, Integer> mp = new HashMap<>();
        mp.put('I', 1);
        mp.put('V', 5);
        mp.put('X', 10);
        mp.put('L', 50);
        mp.put('C', 100);
        mp.put('D', 500);
        mp.put('M', 1000);

        for(int i=0; i<s.length(); i++ ) {
            int curr = mp.get(s.charAt(i));
            int next = ((i + 1) < s.length()) ? mp.get(s.charAt(i + 1)) : 0;

            if(curr < next) {
                result -= curr;
            }

            else {
                result += curr;
            }
        }
        return result;

    }
}
```

### Integer to Roman

```
class Solution {
    public String intToRoman(int num) {
        int nums[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String som[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV","I"};
        
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<nums.length; i++ ) {
            while(nums[i] <= num) {
                sb.append(som[i]);
                num -= nums[i];
            }
        }
        return sb.toString();
    }
}
```

### 2 Sum

### 3 Sum

```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Set<List<Integer>> set = new HashSet<>();

        for(int i=0; i<nums.length; i++ ) {
            Set<Integer> lst = new HashSet<>();
            for(int j=i+1; j<nums.length; j ++) {
                int third = -(nums[i] + nums[j]);
                if(lst.contains(third)) {
                    List<Integer> temp = Arrays.asList(nums[i], nums[j], third);
                    Collections.sort(temp);
                    set.add(temp);
                } 
                lst.add(nums[j]);
            }
        }

        List<List<Integer>> result = new ArrayList<>(set);
        return result;

    }
}
```

