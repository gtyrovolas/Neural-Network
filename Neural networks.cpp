#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include <assert.h>
#include <cmath>
#include <vector>

using namespace std;

typedef long long ll;

const double e = 2.71828182845904523536028747135266249775724709369995;
const double pi = 3.14159265359;
// All these are the infastructure of a matrix
struct mat{ // Matrix for at most 20x5
    ll m, n; // matrix is m x n
    vector< vector<double> > M;
    mat(ll m1, ll n1){
      m = m1;
      n = n1;

      M.resize(m1);
      for(ll i = 0; i < m1; i++){
        M[i].resize(n1);
      }
    }
    mat(){}
    void set(ll m1, ll n1){
      m = m1; n = n1;
      M.resize(m1);
      for(ll i = 0; i < m1; i++){
        M[i].resize(n1);
      }
    }
    mat operator +(const mat &A){ // matrix addition
      if(n != A.n || m != A.m)
      {
        throw invalid_argument( "Addition failed due to wrong dimensions" );
      }
      else
      {
        mat sol(m,n);
        for(ll i = 0; i < m; i++){
          for(ll j = 0; j < n; j++){
            sol.M[i][j] = M[i][j] + A.M[i][j];
          }
        }
        sol.m = m;
        sol.n = n;
        return sol;
      }
    }
    mat operator *(const mat &A){ // matrix multiplication
      if(n != A.m){
        throw invalid_argument( "Multiplication failed due to incompatible matrices" );
      }
      mat sol(m, A.n);
      for(ll i = 0; i < m; i++){
        for(ll j = 0; j < A.n; j++){
          double sum = 0;
          for(ll i2 = 0; i2 < n; i2++){
     //       cout << "I " << i << " j " << j << " i2 "<< i2 << endl;
            sum += M[i][i2] * A.M[i2][j];
          }
          sol.M[i][j] = sum;
        }
      }
      return sol;
    }
    mat operator -(const mat &A){ // subtraction
      if(n != A.n || m != A.m)
      {
        throw invalid_argument( "Subtraction failed due to wrong dimensions" );
      }
      else
      {
        mat sol(m,n);
        for(ll i = 0; i < m; i++){
          for(ll j = 0; j < n; j++){
            sol.M[i][j] = M[i][j] - A.M[i][j];
          }
        }
        return sol;
      }
    }
    mat operator =(const mat &A){ // equals
      M.resize(A.m);
      for(ll i = 0; i < m; i++){
        M[i].resize(A.n);
      }
      m = A.m;
      n = A.n;
      for(ll i = 0; i < m; i++){
        for(ll j = 0; j < n; j++){
          M[i][j] = A.M[i][j];
        }
      }
    }
};
mat T(mat A){
    mat sol(A.n, A.m);

    for(ll i = 0;i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        sol.M[j][i] = A.M[i][j];
      }
    }
    return sol;
}
void out(mat A){ // output Matrix Contents
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        cout << A.M[i][j] << " ";
      }
      cout << endl;
    }
    cout << endl;
}
void outD(mat A){ // ouput Dimensions
    cout << A.m << " " << A.n << endl;
}
mat multiply(mat A, mat B){ // scalar multiplication of two matrices
    if(A.m != B.m || A.n != B.n) throw invalid_argument("Scalar Multiplication of Matrices Failed\n");
    mat sol(A.n,A.m);
    sol.n = A.n;
    sol.m = A.m;
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        sol.M[i][j] = A.M[i][j] * B.M[i][j];
      }
    }
    return sol;
}
double fRand(double fMin, double fMax) // generate random numbers
{

    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
void genMat(mat & A, double fMin, double fMax){ // fill a matrix with elements from -5 to 5
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        A.M[i][j] = fRand(fMin,fMax);
      }
    }
}
mat scalMult(mat A, double sc){ // multiply a matrix A by a scalar
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        A.M[i][j] *= sc;
      }
    }
    return A;
}




// setting HyperParameters
ll inputSize = 1; // number of input neurons
ll outputSize = 1; // number of output neurons
ll hiddenLayer = 1; // number of neurons in the hidden layers
ll testSize = 0;  // number of testing data
ll dataM = 5;  // number of cases, also one dimension of the input matrix
ll dataN = inputSize;  // number of input neurons


mat X(dataM,dataN); // input
mat Y(dataM,outputSize); // result
mat testX; // testing input data
mat testY; // testing output data

mat W[10]; // matrix that contains the weights of the synapses
mat Z[5];  // matrix that contains the raw version of the data
mat A[5]; // matrix that contains the activated data


double sigmoid(double z){ // sigmoid of a number
    return (double)1/(1 + pow(e, -z));
}
mat sigmoid(mat z){ // making the sigmoid of each item in a matrix
    mat sol(z.m, z.n);
    for(ll i = 0; i < z.m; i++){
      for(ll j = 0; j < z.n; j++){
        sol.M[i][j] = (double)1/(1 + pow(e, -(z.M[i][j])));
      }
    }
    return sol;
}

double norm(double in){ // function to normalise complex input
    return in;
}

double denorm(double in){ // function to denormalise complex input
    return in;

}
void init(){ // initialising the training

    // initialising input
    X.m = dataM;
    X.n = dataN;
    for(ll i = 0; i < dataM; i++){
      X.M[i][0] = (pi*i/2)/ (dataM - 1);
    }
    // initialising output
    Y.m = dataM;
    Y.n = outputSize;

    for(ll i = 0; i < dataM; i++){
      Y.M[i][0] = norm(sin(X.M[i][0]));
    }

    // initialising weights
    W[1].set(inputSize,hiddenLayer);
    W[1].m = inputSize;
    W[1].n = hiddenLayer;
    genMat(W[1], -2, 2);
    W[2].set(hiddenLayer,outputSize);
    W[2].m = hiddenLayer;
    W[2].n = outputSize;
    genMat(W[2], -2, 2);

    // initialising testing data

    testX.set(testSize,X.n);
    testX.m = testSize;
    testX.n = X.n;

    testY.set(testSize, Y.n);
    testY.m = testSize;
    testY.n = Y.n;


    for(int i = 0; i < testSize; i++){
      testX.M[i][0] = fRand(0, pi);
      testY.M[i][0] = norm(sin(testX.M[i][0]));
    }

}

mat forward(mat X){  // forward propagation
    mat yHat(dataM, outputSize);
    outD(X);
    outD(W[1]);
    Z[2] = X * W[1];
    cout << "tes2t "<<endl;
    A[2] = sigmoid(Z[2]);
    Z[3] = A[2] * W[2];
    yHat = sigmoid(Z[3]);
    return yHat;
}
double cost(mat X, mat y){ // cost function
    mat yHat = forward(X);
    double error = 0;
    for(ll i = 0; i < dataM; i++){
      error += (y.M[i][0] - yHat.M[i][0]) * (y.M[i][0] - yHat.M[i][0]);
    }
    return error/(2 * dataM);
}
double cost2(mat A, mat B){ // cost function
    double error = 0;
    assert(A.m == B.m && A.n == B.n);
    for(ll i = 0; i < A.m; i++){
      error += (A.M[i][0] - B.M[i][0]) * (A.M[i][0] - B.M[i][0]);
    }
    return error/(2 * A.m);
}
double sigmoidPrime(double z){ // Returns the derivative of the sigmoid function
    double p = pow(e, -z);
    return p/((1 + p)*( 1+ p));
}

mat sigmoidPrime(mat z){ // returns the derivative of each element in a matrix
    mat sol(z.m, z.n);
    sol.n = z.n;
    sol.m = z.m;
    for(ll i = 0; i < sol.m; i++){
      for(ll j = 0; j < sol.n; j++){
        sol.M[i][j] = sigmoidPrime(z.M[i][j]);
      }
    }
    return sol;
}

void minErrorRand(){
    double minim = 100;
    mat idx[3], sol;
    for(ll i = 0; i < 1000000; i++){
      init();
      mat yhat = forward(X);
      double error = cost(yhat, Y);
      if(error < minim){
        minim = error;
        idx[1] = W[1];
        idx[2] = W[2];
        sol = yhat;
      }
    }
}

void costFPrime(mat X, mat y, mat &yHat, mat &dJdW1, mat &dJdW2){
    yHat = forward(X);

    mat delta3 = multiply(yHat-y, sigmoidPrime(Z[3]));
    dJdW2 = T(A[2]) * delta3;

    mat delta2 = multiply((delta3 * T(W[2])),sigmoidPrime(Z[2]));
    dJdW1 = T(X) * delta2;
}

void computeNumericalGradient(mat X, mat Y, mat &numdW1, mat &numdW2){ // Numerical gradient Descent
    mat dJdW1, dJdW2;
    double eps = 0.00001;
    numdW1.set(W[1].n, W[1].m);
    numdW2.set(W[2].n, W[2].m);

    numdW1.n = W[1].n; numdW1.m = W[1].m;
    numdW2.n = W[2].n; numdW2.m = W[2].m;

    for(ll i = 0; i < numdW2.m; i++){
      for(ll j = 0; j < numdW2.n; j++){

        W[2].M[i][j] += eps;
        double loss2 = cost(X, Y);

        W[2].M[i][j] -= 2*eps;
        double loss1 = cost(X, Y);

        numdW2.M[i][j] = (loss2 - loss1)/ (2*eps);
        W[2].M[i][j] += eps;
      }
    }

    for(ll i = 0; i < numdW1.m; i++){
      for(ll j = 0; j < numdW1.n; j++){
        W[1].M[i][j] += eps;
        double loss2 = cost(X, Y);

        W[1].M[i][j] -= 2*eps;
        double loss1 = cost(X, Y);

        numdW1.M[i][j] = (loss2 - loss1)/ (2*eps);
        W[1].M[i][j] += eps;
      }
    }
}


void regulate(mat X, mat y, mat &yHat, double rate){
    mat dJdW1, dJdW2;
    computeNumericalGradient(X, y, dJdW1, dJdW2);

    cout << "1 "<< endl;
    mat prevW [10] = {};
    prevW[1] = W[1];
    prevW[2] = W[2];

    cout << "one " << endl;
    yHat = forward(X);

    double prevC;
    do {
      cout << "test "<< endl;
      prevC = cost2(y, yHat);
      W[1] = W[1] - scalMult(dJdW1, rate);
      W[2] = W[2] - scalMult(dJdW2, rate);
      yHat = forward(X);
    }while(cost2(yHat, y) < prevC);
}

ll train(mat X, mat y, ll trials, double rate){
    mat dJdW1, dJdW2;
    mat best;
    ll cnt = 0;
    mat yHat;
    ll id;
    double mn = 100000;
    for(ll i = 0; i < trials; i++){
      regulate(X, y, yHat,rate);
   //   out(yHat);
      if(cost2(yHat, y) < mn){
        mn = cost2(yHat, y);
        id = i;
      }
      if(i % 100 == 0){
        cout << "Trial " << i << " cost is " << cost2(yHat, y) << endl;
      }
    }
    return id;
   // cout << cost(yHat, y) << " after " << cnt << endl;
   // out(yHat);
}


// check with numerical gradient

void getGradients(mat &dJdW1, mat &dJdW2){ // function to get the Gradients computed by costFPrime
    mat yHat;
    costFPrime(X, Y, yHat, dJdW1, dJdW2);
}

double test(mat & yHat){

    yHat = forward(testX);
    return cost2(yHat, testY);

}



int main(){
    srand(time(NULL));

    mat dJdW1, dJdW2;
    mat numdW1, numdW2;
    init();
    train(X, Y, 20000, 10);
    cout << "yHat\n";
  //  out(forward(X));
    cout  << " vs Y" << endl;
  //  out(Y);
    cout << "cost for Y " << cost2(Y,forward(X)) << endl;

    cout << "Weights \n";
    out(W[1]);
    out(W[2]);
    mat yHat;

    for(ll i = 0; i < 31; i++){
      mat t1;
      t1.n = 1; t1.m = 1;
      t1.M[0][0] =(double) (i) / 10;
      mat res = forward(t1);
      cout << "For: " << t1.M[0][0] << " calculated " << denorm(res.M[0][0]) << " correct " << sin(t1.M[0][0]) << endl;
    }

    cout << "Time for interactive testing!!!\n";
    for(ll i = 0; i < 100; i++){
      cout << "Input a number and the program will calculate its sine\n";
      double t;
      cin >> t;
      mat t1;
      t1.n = 1;
      t1.m = 1;
      t1.M[0][0] = t;
      mat res = forward(t1);
      double fin = denorm(res.M[0][0]);
      cout << "The program found " << fin << " as the answer\n";
      cout << "The correct answer is " << sin(t) << endl;
      cout << "The square of the difference is " << (fin - sin(t)) *(fin - sin(t)) << endl << endl << endl;
    }

}




/*
int main()
{
    srand(time(NULL));
    double minim = 100;
    mat B[3];
    mat bestY;
    ll id = 0;
    for(ll i = 0; i < 50; i++){
        cout << i << endl;
        init();
        train(X, Y, 10000);
        double c = cost(forward(X), Y);
        if(c < minim){
          minim = c;
          id = i;
          B[1] = W[1];
          B[2] = W[2];
          bestY = forward(X);
        }
    }
    out(B[1]);
    out(B[2]);
    out(bestY);
    cout << cost(bestY, Y) << endl;
    cout << id << endl;
    return 0;
}*/
