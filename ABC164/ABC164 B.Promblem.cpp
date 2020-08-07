#include <bits/stdc++.h>
using namespace std;

int main() {
  int A, B, C, D;
	cin >> A >> B  >> C >> D;
     
  int E = A / D ;
  int F = C / B ;
  
  if(A % D > 0){
    E++;
  }
  if(C % B > 0){
    F++;
  }
  
  if(E >= F){
    cout << "Yes" <<endl;
  }
  else{
    cout << "No" <<endl;
  }
  
}
