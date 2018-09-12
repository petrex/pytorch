#ifndef CAFFE2_CORE_STORAGE_H_
#define CAFFE2_CORE_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "caffe2/core/allocator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"

#include <ATen/core/Allocator.h>
#include <ATen/core/Device.h>
#include <ATen/core/DeviceType.h>
#include <ATen/core/intrusive_ptr.h>

namespace caffe2 {

class CAFFE2_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  StorageImpl() = delete;
  StorageImpl(const StorageImpl&) = delete;
  StorageImpl& operator=(const StorageImpl&) = delete;

  // Rule of Five
  StorageImpl(StorageImpl&&) = default;
  ~StorageImpl() = default;
  StorageImpl& operator=(StorageImpl&&) = default;

  StorageImpl(
      TypeMeta data_type,
      int64_t numel,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_type_(data_type),
        data_ptr_(std::move(data_ptr)),
        numel_(numel),
        resizable_(resizable),
        allocator_(allocator) {
    if (numel > 0) {
      CAFFE_ENFORCE(
          data_type.id() != TypeIdentifier::uninitialized(),
          "Constructing a storage with meta of unknown type and non-zero numel");
    }
  }

  explicit StorageImpl(at::DeviceType device_type)
      : StorageImpl(device_type, TypeMeta()) {}
  StorageImpl(at::DeviceType device_type, TypeMeta data_type)
      : StorageImpl(
            data_type,
            0,
            at::DataPtr(nullptr, at::Device(device_type)),
            nullptr,
            true) {}

  void reset() {
    data_ptr_.clear();
    numel_ = 0;
  }

  template <typename T>
  inline bool IsType() const {
    return data_type_.Match<T>();
  }

  void* data() const {
    return data_ptr_.get();
  }

  void* data() {
    return data_ptr_.get();
  }

  at::DataPtr& data_ptr() {
    return data_ptr_;
  }

  const at::DataPtr& data_ptr() const {
    return data_ptr_;
  }

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    std::swap(data_ptr_, data_ptr);
    return std::move(data_ptr);
  };

  void set_dtype(const TypeMeta& data_type) {
    int64_t capacity = numel_ * data_type_.itemsize();
    data_type_ = data_type;
    numel_ = capacity / data_type_.itemsize();
  }

  const TypeMeta& dtype() const {
    return data_type_;
  }

  const at::Allocator* allocator() const {
    return allocator_;
  };
  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }

  size_t capacity() const {
    return numel_ * itemsize();
  }

  int64_t numel() const {
    return numel_;
  }

  // TODO: remove later
  void set_numel(int64_t numel) {
    numel_ = numel;
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  inline size_t itemsize() const {
    return data_type_.itemsize();
  }

  bool resizable() const {
    return resizable_;
  };

  void set_resizable(bool resizable) {
    resizable_ = resizable;
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      const TypeMeta& data_type,
      size_t capacity,
      MemoryDeleter d = nullptr) {
    UniqueStorageShareExternalPointer(
        at::DataPtr(src, src, d, data_ptr_.device()), data_type, capacity);
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      const TypeMeta& data_type,
      size_t capacity) {
    data_type_ = data_type;
    CAFFE_ENFORCE_WITH_CALLER(
        data_type_.id() != TypeIdentifier::uninitialized(),
        "To share with a raw external pointer you need to have meta "
        "already set.");
    data_ptr_ = std::move(data_ptr);
    // NOTE: data_type might change and so it's also possible that capacity
    // might not be divisible by itemsize. There is no way for us to keep track
    // of the exact capacity if we're not explicity storing is. More conrectely
    // capacity() might not return the value that was set here, if itemsize does
    // not evenly divide it.
    numel_ = capacity / data_type_.itemsize();
  }

 private:
  TypeMeta data_type_;
  at::DataPtr data_ptr_;
  int64_t numel_;
  bool resizable_;
  at::Allocator* allocator_;
  // allocator_ takes precedence over StaticContext from device_type_
  // Allocator* allocator_;
  // at::DeviceType device_type_ = CPU;
};

class CAFFE2_API Storage {
 public:
  Storage() {}
  Storage(at::DeviceType device_type)
      : storage_impl_(c10::make_intrusive<StorageImpl>(device_type)) {}
  Storage(at::DeviceType device_type, TypeMeta data_type)
      : storage_impl_(
            c10::make_intrusive<StorageImpl>(device_type, data_type)) {}

  Storage(
      TypeMeta data_type,
      int64_t numel,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            data_type,
            numel,
            std::move(data_ptr),
            allocator,
            resizable)) {}

  void reset() {
    storage_impl_->reset();
  }

  template <typename T>
  inline bool IsType() const {
    return storage_impl_->IsType<T>();
  }

  void* data() const {
    return storage_impl_->data();
  }

  void* data() {
    return storage_impl_->data();
  }

  at::DataPtr& data_ptr() {
    return storage_impl_->data_ptr();
  }

  const at::DataPtr& data_ptr() const {
    return storage_impl_->data_ptr();
  }
  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    return storage_impl_->set_data_ptr(std::move(data_ptr));
  };

  void set_dtype(const TypeMeta& data_type) {
    storage_impl_->set_dtype(data_type);
  }

  const TypeMeta& dtype() const {
    return storage_impl_->dtype();
  }
  size_t capacity() const {
    return storage_impl_->capacity();
  }

  int64_t numel() const {
    return storage_impl_->numel();
  }

  // TODO: remove later
  void set_numel(int64_t numel) {
    storage_impl_->set_numel(numel);
  }

  at::DeviceType device_type() const {
    return storage_impl_->device_type();
  }

  const at::Allocator* allocator() const {
    return storage_impl_->allocator();
  }

  inline size_t itemsize() const {
    return storage_impl_->itemsize();
  }

  inline long use_count() const {
    return storage_impl_.use_count();
  }

  inline bool unique() const {
    return storage_impl_.unique();
  }

  void UniqueStorageShareExternalPointer(
      void* src,
      const TypeMeta& data_type,
      size_t capacity,
      MemoryDeleter d = nullptr) {
    CAFFE_ENFORCE_WITH_CALLER(
        storage_impl_.unique(),
        "UniqueStorageShareExternalPointer can only be called when \
        use_count == 1");
    storage_impl_->UniqueStorageShareExternalPointer(
        src, data_type, capacity, d);
  }

  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      const TypeMeta& data_type,
      size_t capacity) {
    CAFFE_ENFORCE_WITH_CALLER(
        storage_impl_.unique(),
        "UniqueStorageShareExternalPointer can only be called when \
        use_count == 1");
    storage_impl_->UniqueStorageShareExternalPointer(
        std::move(data_ptr), data_type, capacity);
  }

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};

} // namespace caffe2

#endif // CAFFE2_CORE_STORAGE_H_
