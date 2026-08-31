// loadgen.cpp - Composite Load Generator: Constant + Poisson + Sine Wave Burst
// Build: g++ -O2 -std=c++17 loadgen.cpp -lcurl -pthread -o loadgen
// Usage: ./loadgen --url <URL> --duration <sec> --constant-qps <constant> --qps <base> --peak-qps <peak> --period-sec <period> --workers <n> --out <file>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <curl/curl.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std::chrono;

// -------------------- 数据结构 --------------------
struct Task {
    double timestamp;
    int id;
};

struct Result {
    double epoch_time;
    double response_ms;
    double inference_ms;
    int id;
};

// -------------------- 线程安全队列 --------------------
template<typename T>
class ThreadQueue {
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv;
    bool closed = false;
public:
    void push(const T& v) {
        {
            std::lock_guard<std::mutex> lk(m);
            q.push(v);
        }
        cv.notify_one();
    }
    bool pop(T &out) {
        std::unique_lock<std::mutex> lk(m);
        while (q.empty() && !closed) cv.wait(lk);
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop();
        return true;
    }
    void close() {
        {
            std::lock_guard<std::mutex> lk(m);
            closed = true;
        }
        cv.notify_all();
    }
};

// -------------------- CURL --------------------
static size_t write_to_string(void* ptr, size_t size, size_t nmemb, void* userdata) {
    std::string* s = reinterpret_cast<std::string*>(userdata);
    s->append(reinterpret_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

double parse_inference_time(const std::string &body) {
    const char key[] = "Inference time:";
    auto pos = body.find(key);
    if (pos == std::string::npos) return -1.0;
    pos += strlen(key);
    while (pos < body.size() && body[pos] == ' ') pos++;
    return atof(body.c_str() + pos);
}

// -------------------- Worker --------------------
void worker_func(ThreadQueue<Task> &taskq,
                 ThreadQueue<Result> &outq,
                 const std::string &url,
                 int timeout_ms,
                 std::atomic<bool> &stop_flag)
{
    CURL* curl = curl_easy_init();
    if (!curl) {
        // std::cerr << "Failed to initialize CURL in worker thread\n";
        return;
    }
    
    // 设置基本选项
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    
    // 启用连接复用和keep-alive
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 0L);
    
    // 增加连接超时设置
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 5000);
    
    // 设置最大连接数（针对HTTP/1.1持久连接）
    curl_easy_setopt(curl, CURLOPT_MAXCONNECTS, 5L);

    while (!stop_flag.load()) {
        Task t;
        if (!taskq.pop(t)) break;

        std::string response;
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        auto t0 = steady_clock::now();
        CURLcode res = curl_easy_perform(curl);
        auto t1 = steady_clock::now();

        double resp_ms = duration_cast<duration<double, std::milli>>(t1 - t0).count();
        double infer_ms = -1.0;
        
        if (res == CURLE_OK) {
            infer_ms = parse_inference_time(response);
        } else {
            // 记录错误但继续运行
            // std::cerr << "CURL error for request " << t.id 
            //           << ": " << curl_easy_strerror(res) << "\n";
        }

        double epoch =
            duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();

        outq.push(Result{epoch, resp_ms, infer_ms, t.id});
    }
    curl_easy_cleanup(curl);
}

// -------------------- Dispatcher --------------------
void dispatcher_func(const std::vector<double> &timestamps,
                     ThreadQueue<Task> &taskq)
{
    auto start = steady_clock::now();
    int id = 0;

    for (double ts : timestamps) {
        double elapsed = duration_cast<duration<double>>(steady_clock::now() - start).count();
        if (ts > elapsed)
            std::this_thread::sleep_for(duration<double>(ts - elapsed));
        taskq.push(Task{ts, ++id});
    }
    taskq.close();
}

// -------------------- Writer --------------------
void writer_func(ThreadQueue<Result> &outq,
                 const std::string &outfile,
                 std::atomic<bool> &stop_flag)
{
    std::ofstream ofs(outfile);
    ofs << "timestamp,response_ms,inference_ms,request_id\n";
    Result r;
    while (outq.pop(r)) {
        ofs << std::fixed << std::setprecision(6)
            << r.epoch_time << ","
            << r.response_ms << ","
            << r.inference_ms << ","
            << r.id << "\n";
    }
    stop_flag.store(true);
}

// -------------------- 时间戳生成 --------------------

// 生成恒定QPS的负载（均匀分布，固定间隔）
std::vector<double> gen_constant_load(int dur, double constant_qps) {
    std::vector<double> ts;
    if (constant_qps <= 0) return ts;
    
    double interval = 1.0 / constant_qps;
    for (double t = 0; t < dur; t += interval) {
        ts.push_back(t);
    }
    return ts;
}

// 生成复合负载：恒定QPS + 泊松QPS + 正弦波突发
std::vector<double> gen_composite_load(int dur, double constant_qps, double base_qps, double peak_qps, double period, unsigned seed) {
    std::mt19937_64 rng_base(seed);
    std::mt19937_64 rng_burst(seed + 1);
    std::uniform_real_distribution<double> ud(0, 1);
    std::vector<double> ts;
    
    // 1. 恒定负载
    auto ts_constant = gen_constant_load(dur, constant_qps);
    ts.insert(ts.end(), ts_constant.begin(), ts_constant.end());
    
    // 2. 基础泊松流量
    for (int s = 0; s < dur; ++s) {
        std::poisson_distribution<int> pd_base(base_qps);
        int base_count = pd_base(rng_base);
        for (int i = 0; i < base_count; ++i)
            ts.push_back(s + ud(rng_base));
    }
    
    // 3. 正弦波突发流量（从0到peak_qps波动）
    for (int s = 0; s < dur; ++s) {
        double burst_qps = peak_qps * std::max(0.0, sin(2 * M_PI * s / period));
        if (burst_qps > 0) {
            std::poisson_distribution<int> pd_burst(burst_qps);
            int burst_count = pd_burst(rng_burst);
            for (int i = 0; i < burst_count; ++i)
                ts.push_back(s + ud(rng_burst));
        }
    }
    
    // 排序所有时间戳
    std::sort(ts.begin(), ts.end());
    return ts;
}

// -------------------- Main --------------------
int main(int argc, char** argv)
{
    std::string url, outfile = "composite.csv";
    int duration = 10, workers = 4, timeout_ms = 3000;
    double constant_qps = 0.0, base_qps = 0, peak_qps = 0, period = 60.0;
    unsigned seed = 12345;

    for (int i=1;i<argc;i++) {
        std::string a = argv[i];
        if (a=="--url") url=argv[++i];
        else if (a=="--duration") duration=std::stoi(argv[++i]);
        else if (a=="--constant-qps") constant_qps=std::stod(argv[++i]);
        else if (a=="--qps") base_qps=std::stod(argv[++i]);
        else if (a=="--peak-qps") peak_qps=std::stod(argv[++i]);
        else if (a=="--period-sec") period=std::stod(argv[++i]);
        else if (a=="--seed") seed=std::stoul(argv[++i]);
        else if (a=="--workers") workers=std::stoi(argv[++i]);
        else if (a=="--timeout-ms") timeout_ms=std::stoi(argv[++i]);
        else if (a=="--out") outfile=argv[++i];
    }

    if (url.empty()) {
        std::cerr << "Error: --url is required\n";
        return 1;
    }

    // std::cout << "Starting loadgen with " << workers << " workers\n";
    // std::cout << "Target URL: " << url << "\n";
    // std::cout << "Duration: " << duration << " seconds\n";
    // std::cout << "Constant QPS: " << constant_qps << ", Base QPS: " << base_qps 
    //           << ", Peak QPS: " << peak_qps << ", Period: " << period << "s\n";
    // std::cout << "Seed: " << seed << "\n";
    
    curl_global_init(CURL_GLOBAL_ALL);

    // 生成复合负载：恒定QPS + 泊松QPS + 正弦波突发（同一时间叠加）
    std::vector<double> timestamps = gen_composite_load(duration, constant_qps, base_qps, peak_qps, period, seed);
    
    // std::cout << "Total requests: " << timestamps.size() << "\n";
    // std::cout << "Average QPS: " << std::fixed << std::setprecision(2) 
    //           << (timestamps.size() / (double)duration) << "\n";
    // std::cout << "QPS Range: " << (constant_qps + base_qps) << " (min) to " 
    //           << (constant_qps + base_qps + peak_qps) << " (max)\n";

    ThreadQueue<Task> taskq;
    ThreadQueue<Result> outq;
    std::atomic<bool> stop(false);

    std::thread writer(writer_func, std::ref(outq), outfile, std::ref(stop));
    std::vector<std::thread> ws;
    for (int i=0;i<workers;i++)
        ws.emplace_back(worker_func, std::ref(taskq), std::ref(outq),
                        url, timeout_ms, std::ref(stop));

    std::thread disp(dispatcher_func, std::cref(timestamps), std::ref(taskq));

    disp.join();
    for (auto &t: ws) t.join();
    outq.close();
    writer.join();

    curl_global_cleanup();
    return 0;
}
