# infosys prep

---

# Arrays and String

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

```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> mp = new HashMap<>();
        for(int i=0; i<nums.length; i++ ) {
            int second = (target - nums[i]);
            if(mp.containsKey(second)) {
                return new int[] {i, mp.get(second)};
            }
            mp.put(nums[i], i);
        }
        return new int[]{0, 0};
    }
}
```

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

### Group Anagrams
```
class Solution {
    public List<List<String>> groupAnagrams(String[] str) {
        HashMap<String, List<String>> mp = new HashMap<>();

        for(int i=0; i<str.length; i ++ ) {
            char[] ch = str[i].toCharArray();
            Arrays.sort(ch);
            String temp = String.valueOf(ch);

            mp.putIfAbsent(temp, new ArrayList<>());
            mp.get(temp).add(str[i]);
        }
        
        return new ArrayList<>(mp.values());
    }
}
```
### Generate all permutations

```
class Solution {
    public void generate(int[] nums, int index, List<List<Integer>> result) {
        if(index == nums.length) {
            List<Integer> lst = new ArrayList<>();
            for(int i: nums) {
                lst.add(i);
            }
            result.add(lst);
            return;
        }

        for(int i=index; i<nums.length; i++ ) {
            swap(nums, index, i);
            generate(nums, index + 1, result);
            swap(nums, index, i);
        }
    }
    
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public List<List<Integer>> permute(int[] nums) {
        List<Integer> lst = new ArrayList<>();
        List<List<Integer>> result = new ArrayList<>();
        generate(nums, 0, result);
        return result;
    }
}
```
### Generate all the permuatations with n1, n2, n3 ... numbers of each character

```
class Main {
    public static List<String> result = new ArrayList<>();
    public static void generate(int a, int b, int c, String temp) {
        if(a == 0 && b == 0 && c == 0) {
            result.add(temp);
            return;
        }
        
        if(a > 0) generate(a - 1, b, c, temp + "a");
        if(b > 0) generate(a, b - 1, c, temp + "b");
        if(c > 0) generate(a, b, c - 1, temp + "c");
    }
    public static void main(String[] args) {
        String str = "abc";
        generate( 1, 1, 1, "");
        for(String i: result) {
            System.out.println(i);
        }
    }
}
```
# Spiral Matrix

```
class Solution {
    public List<Integer> spiralOrder(int[][] mat) {
        
        int row = mat.length;
        int col = mat[0].length;

        int left = 0;
        int right = col - 1;
        int top = 0;
        int bottom = row - 1;

        List<Integer> result = new ArrayList<>();

        while(left <= right && top <= bottom) {
            for(int i = left; i<=right; i++ ) {
                result.add(mat[top][i]);
            }
            top ++;

            for(int i=top; i<=bottom; i++ ) {
                result.add(mat[i][right]);
            }
            right --;

            if(top <= bottom) {
                for(int i=right; i>=left; i --) {
                    result.add(mat[bottom][i]);
                }
                bottom --;
            }

            if(left <= right) {
                for(int i=bottom; i>=top; i --) {
                    result.add(mat[i][left]);
                } 
            left ++;
            }
        }

        return result;
    }
}
```
### Rotate Array 

```
class Solution {
    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        reverse(nums , 0 , nums.length-1);
        reverse(nums , 0 , k-1);
        reverse(nums , k , nums.length-1);

    }
    public void reverse(int[] nums, int start, int end){
        while(start < end){
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            end--;
            start++;
        }

    }
}
```
# Move all zero to end
```
class Solution {
    public void moveZeroes(int[] nums) {
        int a = 0;
        for(int i=0; i<nums.length; i++ ) {
            if(nums[i] != 0) nums[a ++] = nums[i];
        }

        for(int i=a; i<nums.length; i++ ) nums[i] = 0;
        
    }
}
```
