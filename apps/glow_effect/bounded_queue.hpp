// Minimal blocking producer/consumer queue used by the video pipeline.
#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

namespace app {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

    /// Blocks while full. Returns false if the queue was closed.
    bool push(T item) {
        std::unique_lock<std::mutex> lock(m_);
        notFull_.wait(lock, [&] { return closed_ || q_.size() < capacity_; });
        if (closed_) return false;
        q_.push_back(std::move(item));
        notEmpty_.notify_one();
        return true;
    }

    /// Blocks until an item is available. Returns nullopt once closed and drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(m_);
        notEmpty_.wait(lock, [&] { return closed_ || !q_.empty(); });
        return takeLocked();
    }

    /// Non-blocking pop.
    std::optional<T> tryPop() {
        std::lock_guard<std::mutex> lock(m_);
        return takeLocked();
    }

    /// Wakes everybody; pushes fail afterwards, pops drain what is left.
    void close() {
        std::lock_guard<std::mutex> lock(m_);
        closed_ = true;
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

    /// Like close() but also discards pending items.
    void abort() {
        std::lock_guard<std::mutex> lock(m_);
        closed_ = true;
        q_.clear();
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

private:
    std::optional<T> takeLocked() {
        if (q_.empty()) return std::nullopt;
        T item = std::move(q_.front());
        q_.pop_front();
        notFull_.notify_one();
        return item;
    }

    std::mutex m_;
    std::condition_variable notEmpty_, notFull_;
    std::deque<T> q_;
    std::size_t capacity_;
    bool closed_ = false;
};

}  // namespace app
