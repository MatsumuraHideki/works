#include <bits/stdc++.h>
using namespace std;
 
int main() {
  string s, t;
  cin >> s >> t;
  
  int a, b;
  cin >> a >> b;
  
  string u;
  cin >> u;
  
  if(u==s){
    cout << a-1 << " " << b << endl;
  }
  else{
    cout << a << " " << b-1 << endl;
  }
}
