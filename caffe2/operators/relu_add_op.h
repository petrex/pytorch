#ifndef CAFFE2_OPERATORS_RELUADD_OP_H_
#define CAFFE2_OPERATORS_RELUADD_OP_H_

#include <vector>
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SumReluOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit SumReluOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...){}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
};

template <typename T, class Context>
class SumReluGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit SumReluGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...){}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RELUADD_OP_H_
