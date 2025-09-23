  // Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen.cc
 */

#include "codegen_npuir.h"
#include <cassert>
#include <elf.h>
#include <memory>
#include <ostream>
#include <sstream>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/ascend.h"

#include "arith/pattern_match.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"

  this->PrintIndent();
  this->stream << "hivm.hir.mmadL1";
  if(npuirop.a_transpose || npuirop.b_transpose) {
    this->stream << " {";
    this->stream << (npuirop.a_transpose ? "a_transpose" : "");
    this->stream << (npuirop.a_transpose && npuirop.b_transpose ? ", " : "");
    this->stream << (npuirop.b_transpose ? "b_transpose" : "");
    this->stream << "}";
  }
  this->stream << " ins(%" << type_info[a_buffer->name]->var_id;
  this->stream << ", %" << type_info[b_buffer->name]->var_id;
  this->stream << ", %" << init_c_name;
  this->stream << ", %" << real_m_name;
  this->stream << ", %" << real_k_name; 
  this->stream << ", %" << real_n_name;
  this->stream << " : " << GetMemrefInfo(a_buffer->name);
  this->stream << ", " << GetMemrefInfo(b_buffer->name);
  this->stream << ", i1 index, index, index";
  this->stream << ")";
  this->stream << " outs(%" << type_info[c_bufer->name]->var_id;
  this->stream << " : " << GetMemrefInfo(c->buffer->name); 
  this->stream << ")\n";
}

void CodeGenTileLangNPUIR::BinaryVecOpCodegen(const CallNode* op, std::string opName, std::ostream& os){
  // Generate hivm.hir.vadd for tl.npuir_add.
  // before:
  //   T.npuir_add(T.region(A_VEC[0], 1, 1024), T.region(B_VEC[0], 1, 1024),
  //   T.region(C_VEC[0], 2, 1024))
  // after:
  //   hivm.hir.vadd ins(%9, %15 : memref<1024xf32,
  //   #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>)
  //       outs(%16 : memref<1024xf32, #hivm.address_space<ub>>)
  std::string left_data_name = "", right_data_name = "";
  Array<PrimExpr> buffer_shape0, buffer_shape1;
  auto processImm = [&](std::string &data_name, int arg_id,
                        Array<PrimExpr> &buffer_shape) {
    if(auto intImm = op->args[arg_id].as<IntImm>()) {
        auto immObj = intImm.value();
        data_name = PrintExpr(immObj);
        const CallNode *region_node = op->args[1 - arg_id].as<CallNode>();
        const BufferLoadNode *buffer_load_node = 
            region_node->args[0].as<BufferLoadNode();
        if(intImm.value()->dtype != buffer_load_node->buffer->dtype) {
            data_name = ScalarConvertType(&immObj, buffer_load_node->buffer->dtype);
        }
        std::ostringstream temp;
        PrintTypr(buffer_load_node->buffer->dtype, temp);
        this->type_info[data_name] = new Scalar(data_name, temp.str());
    } else if (auto floatImm = op->args[arg_id].as<FloatImm>()) {
        auto immObj = floatImm.value();
        data_name = PrintExpr(immObj);
        const CallNode *region_node = op->args[1 - arg_id].as<CallNode>();
        const BufferLoadNode *buffer_load_node = 
            region_node->args[0].as<BufferLoadNode>();
        if(FloatImm.value()->dtype != buffer_load_node->buffer->dtype) {
            data_name = ScalarConvertType(&immObj, buffer_load_node->buffer->dtype);
        }
        std::ostringstream temp;
        PrintType(buffer_load_node->buffer_dtype, temp);
        this->type_info[data_name] = new Scalar(data_name, temp.str());
    } else {
        const CallNode *region_node = op->args[arg_id].as<CallNode>();
        buffer_shape = region_node->args[0].as<BufferLoadNode>()->buffer->shape;
        data_name = GenSubviewFromRegion(region_node);
    }
  }; 
  processImm(left_data_name, 0, buffer_shape0);
  processImm(left_data_name, 1, buffer_shape1);
  const CallNode *out_region_node = op->args[2].as<CallNode>();
  String out_data_name = "", out_addr_space = "";
  out_data_name = GenSubviewFromRegion(out_region_node);
  this->PrintIndent();
  this->stream << "hivm.hir.v" << opName;
  this->stream << " ins(" << "\%" <, left_data_name << ", " << "\%"
               << right_data_name;
  this->stream << " : ";
  this->stream << " outs(" << "\%" << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  auto dims = broadcastAttrCodegen(buffer_shape0, buffer_shape1);
  if(dims != ""){
    this->stream << " broadcast" << dims;
  }
  this->stream << "\n";
}
void CodeGenTileLangNPUIR::VisitExpr_(const CallNode *op, std::ostream &os) {
  if (op->op.same_as(Op::Get("tl.npuir_pipe_barrier"))) {
    BarrierCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_set_flag"))) {
    tvm::tl::NpuirSetFlag sync_op (op->args, this->vmap);
    PipeFlagCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_wait_flag"))) {
    tvm::tl::NpuirWaitFlag sync_op (op->args, this->vmap);
    PipeFlagCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block"))) {
    tvm::tl::NpuirSyncBlock sync_op (op->args, this->vmap);
    SyncBlockSetCodegen(sync_op, os);
    SyncBlockWaitCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block_set"))) {
    tvm::tl::NpuirSyncBlockSet sync_op(op->args, this->vmap);
    SyncBlockSetCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block_wait"))) {
    tvm::tl::NpuirSyncBlockWait sync_op(op->args, this->vmap);
    SyncBlockWaitCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.ascend_copy"))) {
    AscendCopyCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_add"))) {
    BinaryVecOpCodegen(op, "add", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_exp"))) {
    VexpCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_load_nd2nz"))) {
    Nd2NzCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_store_fixpipe"))) {
    FixpipeCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_dot"))) {
    DotCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_div"))) {
    BinaryVecOpCodegen(op, "div", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_mul"))) {
    BinaryVecOpCodegen(op, "mul", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_sub"))) {
    BinaryVecOpCodegen(op, "sub", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_max"))) {
    BinaryVecOpCodegen(op, "max", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_min"))) {
    BinaryVecOpCodegen(op, "min", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_brc"))) {
    VbrcCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_cast"))) {
    VcastCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_reduce"))) {
    VreduceCodegen(op, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}
void CodeGenTileLangNPUIR::VisitStmt_(const LetStmtNode *op) {
    std::string value = PrintExpr(op->value);
    PrintIndent();
    this->stream << '%' << AllocVarID(op->var.get()) << " = " << value << "\n";
    PrintStmt(op->body);
}

void CodeGenTileLangNPUIR::VisitStmt_(const AttrStmtNode *op) {
    if(op->attr_key == "thread_extent") {
        IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "blockIdx.x" && iv->var->name_hint != "_") {
      this->block_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "%" << this->block_id_
                   << "_i64 = hivm.hir.get_block_idx -> i64\n";
      this->PrintIndent();
      this->stream << "%" << this->block_id_ << " = arith.trunci %"
                   << this->block_id_ << "_i64 : i64 to i32\n";
      this->core_num_ = op->value.as<IntImmNode>()->value;
      } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      auto vec_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "%" << vec_id_
                   << "_i64 = hivm.hir.get_sub_block_idx -> i64\n";
      this->PrintIndent();
      this->stream << "%" << vec_id_ << " = arith.trunci %" << vec_id_
                   << "_i64 : i64 to i32\n";
      }
      this->VisitStmt(op->body);
      return;
    } else if (op->attr_key == "resource_scope") {
      auto resource_id = Downcast<IntImm>(op->value)->value;
      auto resource_name = resource_id == 0 ? "AIC" : "AIV";

      if (NPU_CORETYPE_STR[this->current_coretype] == resource_name) {
      this->VisitStmt(op->body);
      }
      // else do nothing but return.
      return;
    }
    CodeGenC::VisitStmt_(op);
}

/// Generate memref.alloc for TIR AllocateNode like T.decl_buffer.
/// before:
///      A_VEC = T.decl_buffer((128, 256), "float16", scope="shared")
/// after:
///      %A_VEC = memref.alloc() : memref<128x256xf16,
///      #hivm.address_space<ub>>
void CodeGenTileLangNPUIR::VisitStmt_(const ALlocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string scope = GetPtrStorageScope(op->buffer_var);
  std::map<std::string, NPU_CORETYPE> scope_coretype_map{
      {"shared", NPU_CORETYPE::AIV},
      {"shared.dyn", NPU_CORETYPE::AIC},
      {"wmma.accumulator", NPU_CORETYPE::AIC}};
  if (scope_coretype_map[scope] == this->current_coretype) {
    std::string vid = AllocVarID(op->buffer_var.get());
    String address_space = GetAddressSpace(scope);
    // add new memref obj
    this->type_info[vid] =
        new Memref(vid, op->extents, op->dtype, address_space, false);
    this->PrintIndent();
    stream << "%" << vid << " = " << "memref.alloc() : " << GetMemrefInfo(vid)
           << "\n";
  }
  this->VisitStmt(op->body);
}

void CodeGenTileLangNPUIR::VisitStmt_(const MinNode *op, syd::ostream& os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "minsi", os, this);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "minui", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "minnumf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitStmt_(const MaxNode *op, syd::ostream& os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "maxsi", os, this);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "maxui", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "maxnumf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitStmt_(const AddNodeNode *op, syd::ostream& os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "addi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "addi", os, this);
  } 
}

void CodeGenTileLangNPUIR::VisitStmt_(const SubNode *op, syd::ostream& os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "subi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "subi", os, this);
  } 
}

void CodeGenTileLangNPUIR::VisitExpr_(const FloatImmNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::ostringstream temp;
  if (op->value == -std::numeric_limits<float>::infinity()) {
    temp << "arith.constant 0xFF800000 : ";
  } else if (op->value == std::numeric_limits<float>::infinity()) {
    temp << "arith.constant 0x7F800000 : ";
  } else {
    temp << "arith.constant " << op->value << " : ";
  }
  PrintType(op->dtype, temp);
  os << SSAGetID(temp.str(), op->dtype);
}

void CodeGenTileLangNPUIR::VisitExpr_(const IntImmNode *op, std::ostream &os) {
  std::ostringstream temp;
  temp << "arith.constant ";
  if (op->dtype.is_bool()) {
    temp << (op->value == 1 ? "true" : "false");
  } else {
    temp << op->value << " : ";
    PrintType(op->dtype, temp);
  }
  os << SSAGetID(temp.str(), op->dtype);
}

void CodeGenTileLangNPUIR::VisitExpr_(const MulNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "muli", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "mulf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const AndNode *op, std::ostream &os) {
  assert(op->a.dtype().is_int() || op->a.dtype().is_uint());
  assert(op->b.dtype().is_int() || op->b.dtype().is_uint());
  PrintBinary(op, "andi", os, this);
}

void CodeGenTileLangNPUIR::VisitExpr_(const OrNode *op, std::ostream &os) {
  assert(op->a.dtype().is_int() || op->a.dtype().is_uint());
  assert(op->b.dtype().is_int() || op->b.dtype().is_uint());
  PrintBinary(op, "ori", os, this);
}

void CodeGenTileLangNPUIR::VisitExpr_(const DivNode *op, std::ostream &os) {
  PrintBinary(op, "<<<divf>>>", os, this);
}

void CodeGenTileLangNPUIR::VisitExpr_(const SelectNode *op, std::ostream &os) {
  auto condition = PrintExpr(op->condition);
  auto true_value = PrintExpr(op->true_value);
  auto false_value = PrintExpr(op->false_value);

  os << "(" << condition << " ? "
     << "" << true_value << " : " << false_value << ")";
}

voud PrintHostFunc(const PrimFunc &f, const std::string &name, std::string &os,
                   int core){
  os << "extern \"C\" void call(";
  std::vector<std::string> arg_names;
  for (size_t i = 0; i < f->params.size(); ++i) {
    auto v = f->params[i];
    if (i != 0) {
      os << ", ";
    }
    arg_names.push_back(v->name_hint);
    os << "uint8_t* " << v->name_hint;
  }
  os << ", aclrtStream stream) {\n  ";

  os << name << "<<<" << core << ", nullptr, stream>>>(";
  for (auto &arg_name : arg_names) {
    os << arg_name;
    if (arg_name != arg_names.back()) {
      os << ", ";
    }
  }
  os << ");\n";
  os << "}\n";
}

void CodeGenTileLangNPUIR::GenRecastFromArg(Buffer curr_buffer, String arg_name,
                                            String &recast_inst) {
  // reinterpret_cast memref from 1D to xD
  std::ostringstream res;
  String target_strides = GetBufferStrides(curr_buffer);
  String cast_name = arg_name + "_Recast";
  // add new memref obj
  this->type_info[cast_name] = new Memref(cast_name, curr_buffer);
  res << "\%" << cast_name << " = ";
  res << "memref.reinterpret_cast \%";
  res << arg_name;
  res << " to offset: [0], sizes: [";
  PrintShape(curr_buffer->shape, ", ", res);
  res << "], strides: ";
  res << target_strides;
  res << " : ";
  res << GetMemrefInfo(arg_name);
  res << " to ";
  res << GetMemrefInfo(cast_name);
  res << "\n";
  recast_inst = res.str();
}

void CodeGenTileLangNPUIR::AddFunctionForCoreType(const GlobalVal &gvar,
                                                const PrimFunc &f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);

  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  int func_scope = this->BeginScope();
  this->PrintIndent();
  auto func_name = static_cast<std::string>(global_symbol.value());
  if (this->func_coretype == NPU_CORETYPE::MIX) {
    func_name = func_name + "_mix_" + NPU_CORETYPE_str[this->current_coretype];
  }

  this->stream << "func.func @" << func_name << "(";

  std::vector<String> recast_need_insert;
  // add args before memref
  stream << "\%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, "
            "\%arg1: memref<?xi8>, \%arg2: memref<?xi8>, ";
  this->type_info.clear();
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());

    if (i != 0)
      stream << ", ";

    if (v.dtype().is_handle()) {
      this->vmap = f->buffer_map;
      auto real_v = f->buffer_map[v]->data;
      String arg_name = AllocVarID(real_v.get());
      // add new memref obj
      Memref *buffer = new Memref(arg_name, f->buffer_map[v], true);
      this->type_info[arg_name] = buffer;
      stream << "%" << arg_name << ": " << GetMemrefInfo(arg_name);
      String recast_inst = "";
      GenRecastFromArg(f->buffer_map[v], arg_name, recast_inst);
      recast_need_insert.push_back(recast_inst);

      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    } else {
      stream << "%" << vid << ": ";
      CodeGenC::PrintType(GetType(v), stream);
    }
  }
  // add args after memref
  stream << ", \%arg13: i32, \%arg14: i32, \%arg15: i32, \%arg16: i32, "
            "\%arg17: i32, \%arg18: i32";
  stream << ")\n";
  this->PrintIndent();
  stream
      << "attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : "
         "i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, "
         "hivm.func_core_type = #hivm.func_core_type<"
      << NPU_CORETYPE_STR[this->current_coretype] << ">";
  if (this->func_coretype == NPU_CORETYPE::MIX) {
    stream << ", hivm.part_of_mix, mix_mode = \"mix\"";
  }
  stream << "} {\n";

  int func_body_scope = this->BeginScope();
  this->PrintIndent();
  stream << "hivm.hir.set_ffts_base_addr \%arg0\n";
  for (String recast_inst : recast_need_insert) {
    this->PrintIndent();
    stream << recast_inst;
  }
  this->PrintStmt(f->body);
  this->EndScope(func_body_scope);
  this->PrintIndent();
  this->stream << "return\n";
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(func_scope);
}

/// Infer function core type: aic, aiv, mix
class InferFuncCoreType : public StmtExprVisitor {
  std::map<std::string, NPU_CORETYPE> scope_coretype_map{
      {"shared", NPU_CORETYPE::AIV},
      {"shared.dyn", NPU_CORETYPE::AIC},
      {"wmma.accumulator", NPU_CORETYPE::AIC},
      {"wmma.matrix_a", NPU_CORETYPE::AIC},
      {"wmma.matrix_b", NPU_CORETYPE::AIC}};

public:
  void VisitStmt(const Stmt &stmt) override {
    StmtExprVisitor::VisitStmt(stmt);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    // It is mixkernel iff there exists T.rs.
    if (op->attr_key == "resource_scope") {
      func_coretype = NPU_CORETYPE::MIX;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    // It is cube kernel if there exists T.npuir_dot.
    if (op->op.same_as(Op::Get("tl.npuir_dot"))) {
      if (func_coretype != NPU_CORETYPE::MIX) {
        func_coretype = NPU_CORETYPE::AIC;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const AllocateNode *op) final {
    // It is cube kernel if there exists buffer with shared.dyn/wmma.xxx address
    // space
    std::string scope = GetPtrStorageScope(op->buffer_var);
    if (func_coretype != NPU_CORETYPE::MIX) {
      if (scope_coretype_map.count(scope) != 0) {
        func_coretype = scope_coretype_map[scope];
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  NPU_CORETYPE func_coretype{NPU_CORETYPE::AIV};
};

void CodeGenTileLangNPUIR::AddFunction(const GlobalVar &gvar,
                                       const PrimFunc &f) {
  InferFuncCoreType infer;
  infer.VisitStmt(f->body);

  this->func_coretype = infer.func_coretype; // NPU_CORETYPE::MIX;

  this->stream
      << "module attributes {hivm.module_core_type = #hivm.module_core_type<"
      << NPU_CORETYPE_STR[this->func_coretype] << ">} {\n";

  if (this->func_coretype == NPU_CORETYPE::MIX ||
      this->func_coretype == NPU_CORETYPE::AIC) {
    this->current_coretype = NPU_CORETYPE::AIC;
    AddFunctionForCoreType(gvar, f);
  }

  if (this->func_coretype == NPU_CORETYPE::MIX ||
      this->func_coretype == NPU_CORETYPE::AIV) {
    this->current_coretype = NPU_CORETYPE::AIV;
    AddFunctionForCoreType(gvar, f);
  }

  this->PrintIndent();
  this->stream << "}\n\n";

  //  PrintHostFunc(f, func_name, stream, this->core_num_);
}

String CodeGenTileLangNPUIR::GetMemrefInfo(String name) {
  if (this->type_info.count(name) == 0)
    LOG(FATAL) << "Can not find memref ssa object: " << name;
  if (!dynamic_cast<Memref *>(type_info[name])) {
    LOG(FATAL) << name << " should be a memref";
  }
  Memref *MemrefObj = dynamic_cast<Memref *>(this->type_info[name]);
  return GetMemrefInfo(MemrefObj);
}

String CodeGenTileLangNPUIR::GetMemrefInfo(Memref *memrefObj) {
  if (memrefObj->type_str != "")
    return memrefObj->type_str;
  std::ostringstream memref_type;
  memref_type << "memref<";
  // dump shape
  if (memrefObj->is_arg) {
    memref_type << "?x";
  } else {
    for (PrimExpr s : memrefObj->shape) {
      if (auto s_int = as_const_int(s)) {
        memref_type << std::to_string(*s_int) + "x";
      } else {
        // not support ssa var in memref size
        memref_type << "?x";
      }
    }
  }
  PrintType(memrefObj->dtype, memref_type);
  // dump strides and offset
  if (!memrefObj->is_arg) {
    memref_type << ", strided<[";
    for (int i = 0; i < memrefObj->dim; i++) {
      if (i > 0)
        memref_type << ", ";
      if (memrefObj->stride.size() > 0) {
        if (auto s_int = as_const_int(memrefObj->stride[i])) {
          memref_type << std::to_string(*s_int);
        } else {
          // not support ssa var in memref size
          memref_type << "?";
        }
      } else {
        memref_type << memrefObj->stride_int[i];
      }
    }
    memref_type << "], offset:";
    if (memrefObj->var_offset)
      memref_type << "?";
    else
      memref_type << memrefObj->offset;
    memref_type << ">";
  }
  memref_type << ", #hivm.address_space<" << memrefObj->address_space << ">>";
  memrefObj->type_str = memref_type.str();
  return memrefObj->type_str;
}

void Memref::GetIntStride() {
  if (stride.empty()) {
    stride_int = GetStrideFromShape(shape);
    for (unsigned long s : stride_int) {
      stride.push_back(IntImm(DataType::Int(64), s));
    }
  } else {
    for (PrimExpr s : stride) {
      if (auto s_int = as_const_int(s))
        stride_int.push_back(*s_int);
    }
  }
}

Memref::Memref(String name, Array<PrimExpr> shape_in, DataType dtype_in,
               String addr_space_in, bool var_offset_in,
               Array<PrimExpr> stride_in, int offset_in, bool is_arg_in) {
  var_id = name;
  shape = shape_in;
  stride = stride_in;
  dtype = dtype_in;
  offset = offset_in;
  is_arg = is_arg_in;
  address_space = addr_space_in;
  var_offset = var_offset_in;
  dim = shape_in.size();
  GetIntStride();
}

Memref::Memref(String name, Buffer buffer, bool is_arg_in) {
  var_id = name;
  shape = buffer->shape;
  stride = buffer->strides;
  dtype = buffer->dtype;
  offset = 0;
  is_arg = is_arg_in;
  String scope = GetPtrStorageScope(buffer->data);
  address_space = GetAddressSpace(scope);
  var_offset = false;
  dim = shape.size();
  GetIntStride();
}

} // namespace codegen
} // namespace tvm