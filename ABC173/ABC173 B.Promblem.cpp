#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n;
  cin >> n;
  
  vector<string> s(n);
  for(int i=0; i<n; i++){
    cin >> s[i];
  }
  
  vector<int> a(4);
  for(int j=0; j<n; j++){
    if(s[j] == "AC"){
      a[0]++;
    }
    else if(s[j] == "WA"){
      a[1]++;
    }
    else if(s[j] == "TLE"){
      a[2]++;
    }
    else{
      a[3]++;
    }
  }
  
    cout << "AC x " << a[0] << endl;
    cout << "WA x " << a[1] << endl;
    cout << "TLE x " << a[2] << endl;
    cout << "RE x " << a[3] << endl;
}
