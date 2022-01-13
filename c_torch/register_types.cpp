#include "register_types.h"

#include "core/class_db.h"
#include "c_torch.h"

void register_c_torch_types() {
    ClassDB::register_class<CTorch>();
}

void unregister_c_torch_types() {
   // Nothing to do here in this example.
}