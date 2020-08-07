#include <bits/stdc++.h>
using namespace std;

int main(){
  char ac;
  cin >> ac;
  
  
  string cap={"abcdefghijklmnopqrstuvwxyz"};
  string CAP={"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
  
  for(int i=0; i<26; i++){
  if(ac==cap[i]){
    cout << 'a' << endl;
  }
  else if(ac==CAP[i]){
    cout << 'A' << endl;
  }
  }
}
