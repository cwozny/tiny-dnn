/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

// this example shows how to use tiny-dnn library to fit data, by learning a
// sinus function.

// please also see:
// https://github.com/tiny-dnn/tiny-dnn/blob/master/docs/how_tos/How-Tos.md

#include "INTEGRITY.h"

#include <cstdio>

#include "tiny_dnn/tiny_dnn.h"

int main(int argc, char* argv[])
{
  tiny_dnn::network<tiny_dnn::sequential> net;

  net << tiny_dnn::fully_connected_layer<tiny_dnn::tan_h>(1, 10);
  net << tiny_dnn::fully_connected_layer<tiny_dnn::tan_h>(10, 10);
  net << tiny_dnn::fully_connected_layer<tiny_dnn::tan_h>(10, 1);
  
  // create input and desired output on a period
  std::vector<tiny_dnn::vec_t> X;
  std::vector<tiny_dnn::vec_t> sinX;

  for (double x = -M_PI; x <= M_PI; x += 0.1)
  {
    tiny_dnn::vec_t vx    = {x};
    tiny_dnn::vec_t vsinx = {sin(x)};

    X.push_back(vx);
    sinX.push_back(vsinx);
  }

  // set learning parameters
  size_t batch_size = 16;    // 16 samples for each network weight update
  int epochs        = 2000;  // 2000 presentation of all samples
  tiny_dnn::adam opt;

  // this lambda function will be called after each epoch
  int iEpoch              = 0;
  auto on_enumerate_epoch = [&]()
  {
    // compute loss and disp 1/100 of the time
    iEpoch++;
    if (iEpoch % 100) return;

    double loss = net.get_loss<tiny_dnn::mse>(X, sinX);

    printf("epoch=%d/%d loss=%f\n", iEpoch, epochs, loss);
  };

  // learn
  printf("learning the sin function with 2000 epochs:\n");
  net.fit<tiny_dnn::mse>(opt, X, sinX, batch_size, epochs, []() {},
                         on_enumerate_epoch);

  printf("\nTraining finished, now computing prediction results:\n");

  // compare prediction and desired output
  double fMaxError = 0.0;

  for (double x = -M_PI; x <= M_PI; x += 0.1)
  {
    tiny_dnn::vec_t xv = {x};
    double fPredicted   = net.predict(xv)[0];
    double fDesired     = sin(x);

    printf("sin(%f) = %f\tpredicted = %f\terror = %f\n", x, fDesired, fPredicted, (fPredicted-fDesired));

    // update max error
    double fError = abs(fPredicted - fDesired);

    if (fMaxError < fError) fMaxError = fError;
  }

  printf("max_error = %f\n", fMaxError);
  
  return 0;
}
