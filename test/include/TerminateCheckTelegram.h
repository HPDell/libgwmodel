#include <string>
#include <mutex>
#include <gwmodel.h>
#include <queue>
#include <thread>
#include <condition_variable>

using namespace gwm;

class TerminateCheckTelegram : public Logger
{
public:
    static std::queue<std::string> MessageQueue;

#ifdef ENABLE_OPENMP
    static std::mutex Lock;
    static bool TerminatePrinter;
#endif

    static void progress_print();

    TerminateCheckTelegram(std::string breakStage, std::size_t breakProgress) :
#ifdef ENABLE_OPENMP
        mPrintThread(progress_print),
#endif
        mBreakStage(breakStage),
        mBreakProgress(breakProgress)
    {}

    ~TerminateCheckTelegram()
    {
#ifdef ENABLE_OPENMP
        TerminatePrinter = true;
        mPrintThread.join();
#endif
    }

    bool stop() override { return mCancelled; }

    void progress(std::size_t current, std::size_t total, std::string fun_name, std::string file_name) override;

    void progress(double percent, std::string fun_name, std::string file_name) override;

private:
    bool mCancelled = false;
    std::string mBreakStage = "";
    std::size_t mBreakProgress = 0;

#ifdef ENABLE_OPENMP
    std::thread mPrintThread;
    std::condition_variable mPrintable;
#endif

};

const std::map<ParallelType, std::string> ParallelTypeDict = {
    std::make_pair(ParallelType::SerialOnly, "SerialOnly"),
    std::make_pair(ParallelType::OpenMP, "OpenMP"),
    std::make_pair(ParallelType::CUDA, "CUDA"),
    std::make_pair(ParallelType::MPI, "MPI"),
    std::make_pair(ParallelType::MPI_Serial, "MPI_Serial"),
    std::make_pair(ParallelType::MPI_MP, "MPI_MP"),
    std::make_pair(ParallelType::MPI_CUDA, "MPI_CUDA")
};
