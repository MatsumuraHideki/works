#include <bits/stdc++.h>
using namespace std;
 
int main() {
  string s, t;
  cin >> s >> t;
  
  if(t.size()==s.size()+1){
  
    int sum = 0;
  for(int i=0;i<s.size();i++){
    if(s.at(i)==t.at(i)){
   sum++;
    }
  }
  
    if(sum==s.size()){
      cout << "Yes" << endl;
    }
    else{
    cout << "No" << endl;
    }
  }
  else{
    cout << "No" << endl;
  }
}
