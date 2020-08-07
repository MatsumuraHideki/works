#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  
  if(n / 100 == 7){
    cout << "Yes" << endl;
  }
  
  else{
    if( (n % 100) / 10 == 7){
    cout << "Yes" << endl;
  }
    else{
      if( (n % 10)  == 7){
    cout << "Yes" << endl;
  }
  else{
    cout << "No" << endl;
  }
      
    }
  }
}
