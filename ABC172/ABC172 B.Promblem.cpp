#include <bits/stdc++.h>
using namespace std;
 
int main() {
  string s, t;
  cin >> s >> t;
  
  int n=s.size();
  int num=0;
  for(int i=0; i<n; i++){
    if(s[i] != t[i]){
      num++;
    }
  }
  
  cout << num << endl;
}
