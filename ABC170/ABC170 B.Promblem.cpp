#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int x, y;
  cin >> x >> y;
 
  if(y%2==1){
    cout << "No" << endl; 
  }
  else if(y<2*x){
    cout << "No" << endl; 
  }
  else if(4*x < y){
    cout << "No" << endl; 
  }
  else{
    cout << "Yes" << endl; 
  }
  
}
