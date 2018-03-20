#include "add_3_ctx.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "TESTS/test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>


class AddTest3 : public Test {
    Context ctx;
    TensorIdxImporter t_import;
public:
    void runAll(void);
};

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char* argv[]) {
    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    AddTest3 test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

    return 0;
}

void AddTest3::runAll(void) {
    testStart("simple add test 3");
    timer_start();
    get_test_quant_add_3_ctx(ctx);
    S_TENSOR z = ctx.get("z:0");
    Tensor* ptr_z = z.get();
    ctx.eval();
    timer_stop();

    Tensor* ref_z = t_import.float_import("/fs/idx_data/output_z.idx");

    // compare the results
    double result = meanPercentErr<float>(ref_z, ptr_z);
    passed(result < 0.001);
}