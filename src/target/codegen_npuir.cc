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
#include "tvm/runtime/data_tpye.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace codegen {

static std::map<NPU_CORETYPE, std::string> NPU_CORETYPE_STR{
    {NPU_CORETYPE::AIC, "AIC"},
    {NPU_CORETYPE::AIV, "AIV"},
    {NPU_CORETYPE::MIX, "MIX"}};

static std::map<NPU_CORETYPE, std::string> NPU_CORETYPE_str{
    {NPU_CORETYPE::AIC, "aic"},
    {NPU_CORETYPE::AIV, "aiv"},
    {NPU_CORETYPE::MIX, "mix"}};

static std::map<NPU_CORETYPE, std::string> coretype_syncblock_map {
    {NPU_CORETYPE::AIC, "CUBE"},
    {NPU_CORETYPE::AIV, "VECTOR"}};

static std::map<int, std::string> fixpipe_pre_relu_mode{
    {0, "NO_RELU"}, {1, "NORMAL_RELU"}, {2, "LEAKY_RELU"}, {3 "P_RELU"}};

static std::map<tl::SyncBlockMode, std::string> SyncBlockMode_str{
    {tl::SyncBlockMode::INTER_BLOCK, "INTER_BLOCK_SYNCHRONIZATION"},
    {tl::SyncBlockMode::INTER_SUBBLOCK, "INTER_SUBBLOCK_SYNCHRONIZATION"},
    {tl::SyncBlockMode::INTER_SUBBLOCK, "INTRA_SUBBLOCK_SYNCHRONIZATION"}};

constexpr uint8_t FLAG_ID_BITS = 64;

template <typename T>
inline void PrintBinary(const T *op, const char *opstr, std::ostream &os,
                        CodeGenC *CG) {
  auto PrintOp = [op, &os, CG](auto Operand) {
    std::ostringstream tmpos;
      if (Operand.template as <tvm::tir::IntImmNode)() ||
          Operand.template as <tvm::tir::VarNode() {
        CG->PrintExpr(Operand, tmpos << "%");
      } else if (auto *float_imm =
                      Operand.template as<tvm::tir::FloatImmNode>()) {
        tmpos << "Invalid float scalar operation"
              << "\n";
      } else {
        // TOD): codegen expr as a seperate instruction in mlir
        tmpos << "<<<expr:%" << Operand << ">>>";
      }
      return tmpos.str();
  };
  if (op->dtype.lanes() == 1) {
    // left op
    os << "arith." << opstr << " ";
    os << PrintOp(op->a);
    os << " : ";
    CG->PrintType(op->a->dtype, os);
  } else {
    os << "<<<invalid-op-dtype-lanes-not-one: %" << opstr << ">>>\n";
  }
}

String GetAddressSpace(String address_space) {
  if (address_space == "global")
    return "gm";
  else if (address_space == "shared")
    return "ub";
  else if (address_space == "shared.dyn")
    return "cbuf";
  else if (address_space == "wmma.accumulator")
    return "cc";
  return "unknown";
}

bool IsEqual(Array<PrimExpr> a, Array<PrimExpr> b) {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (!(a[i].same_as(b[i])))
      return false;
  }
  return true;
}

bool AllZero(Array<PrimExpr> a) {
  for (PrimExpr pe: a) {
    if (!is_zero(pe))
      return false;
  }
  return true;
}

std::vector<unsigned long> GetStrideFromShape(Array<tvm::PrimExpr shape) {
  std::vector<unsigned long> strides;
  unsigned long total_size = 1;
  std::vector<int> shape_int;
  for (PrimExpr s : shape) {
    if (auto s_int = as_const_int(s)) {
        total_size *= *s_int;
        shape_int.push_back(*s_int);
    }
  }
  for (int i = 0; i < shape.size(); i++) {
    total_size /= shape_int[i];
    strides.push_back(total_size);
  }
  return strides;
}

// get string formate of buffer stride
// If the buffer stride is empty, indicating buffer is contiguous
String GetBufferStrides(Buffer buffer) {
  Array<PrimExpr> shape = buffer->shape;
  std::vector<unsigned long> strides;
  int dim = buffer->shape.size();
  if (buffer->strides.empty()) {
      strides = GetStrideFromShape(shape);
  } else {
    for (PrimExpr stride : buffer->strides) {
      if (auto stride_int = as_const_int(stride)) {
        strides.push_back(*stride_int);
      }
    }
  }
  Atring res = "[";
  for (int i = 0; i < dim; i++) {
    if (i > 0)
      res = res + ", ";
    res = res + std::to_string(strides[i]);
  }
  res = res + "]";
  return res;
}
static std::vector<int> getBroadcastDim(Array<PrimExpr> &buffer_shape0,
                                        Array<PrimExpr> &buffer_shape1) {
  assert(buffer_shape0.size() == buffer_shape1.size());
  std::vector<int> dims;
  for (int i = 0; i < buffer_shape0.size(); i++) {
    if (*as_const_int(buffer_shape0[i]) == 1 &&
        *as_const_int(buffer_shape1[i]) != 1) {
        dims.emplace_back(i);
    }
    if (*as_const_int(buffer_shape0[i]) != 1 &&
        *as_const_int(buffer_shape1[i]) == 1) {
        dims.emplace_back(i);
    }
    assert(*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i]))
  }
  return dims;
}

static std::string broadcastAttrCodegen(Array<PrimExpr> &buffer_shape0,
                                        Array<PrimExpr> &buffer_shape1) {
  if (buffer_shape0.empty() || buffer_shape1.empty()) {
    return "";
  }
  std::vector<int> broadcastDims;
  if (buffer_shape0.size() && buffer_shape1.size()) {
    broadcastDims = getBroadcastDim(buffer_shape0, buffer_shape1);
  }
  std::ostringstream temp;
  if (broadcastDims.size()) {
    temp << " = [";
    for (auto dim: broadcastDims) {
      temp << dim;
      if (dim != broadcastDims.back()) {
        temp << ", ";
      }
    }
    temp << "]";
  }
  return temp.str();
}

template <typename T>
std::string CodeGenTileLangNPUIR::ScalarConvertType(T *imm
                                                    DataType targetDtype) {
  auto castNote = std::make_unique<tir::Cast>(targetDtype, *imm);
  std::string castId = SSAGetID(PrintExpr(*castNode), targetDtype);
  return castId;
}

CodeGenTileLangNPUIR::CodeGenTileLangNPUIR() {
  // restrict_keyword_ = "GM_ADDR";
}

void CodeGenTileLangNPUIR::PrintFuncPrefix(std::ostream &os) {
  // os << "CATLASS_GLOBAL\n";
}

std::string CodeGenTileLangNPUIR::Finish() {
  // decl_stream << "#include \"common.h\"\n";
  // decl_stream << "#include \"acl/acl.h\"\n";
  // decl_stream << "using namespace Catlass;\n";
  // decl_stream << "\n";
  std:ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str()
}

void CodeGenTileLangNPUIR::VisitStmt_(const tir::ForNode *op) {
  auto flush == false;
  if (flush_out_) {
    flush = true;
    flush_out_ = false;
  }
  // TODO: Do we need add unroll attribute?
  // if (op->kind == tir::ForKind::kUnrolled) {
  //     PrintIndent();
  //     stream << "#pragma unroll\n";
  // }
  std::string upperBoundId = 
      SSAGetID(PrintExpr(arith::Analyzer().Simplify(op->extent + op->min)),
               op->extent->dtype);
  assert(op->exten.dtype().is_int() || op->extent.dtype().is_uint());
  assert(op->min.dtype() == op->extent.dtype());
  std::string vid = 
      SSAGetID(AllocVarID(op->loop_var.get()), op->loop_var->dtype);
  std::string lowerBoundId = SSAGetID(PrintExpr(op->min), op->min->dtype);
  std::string extentId = SSAGetID(PrintExpr(op->extent), op->extent->dtype);
  auto stepNode = std::make_unique<IntImm>(op->min.dtype(), 1);
  auto stepId = SSAGetID(PrintExpr(*stepNode), stepNode->dtype());
  PrintIndent();
  stream << "scf.for %" << vid << "= %" << lowerBoundId << " to %"
         << upperBoundId << " step %" << stepId << " : ";
  PrintType(op->min.dtype(), stream);
  stream << " {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangNPUIR::VisitStmt_(const tir::IfThenElseNode *op) {
  std::string cond == SSAGetID(PrintExpr(op->condition), op->condition->dtype);
  PrintIndent();
  stream << "scf.if %" << cond << " {\n}";
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);
  if (op->else_case) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope  = BeginScope();
    PrintStmt(op->else_case.value());
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangNPUIR::PrintSSAAssign(const std::string &target,
                                          const std::string &src, DataType t) {
  stream << "%" << target << " = " << src << "\n";
}

void CodeGenTileLangNPUIR::PrintShape(Array<PrimExpr> shape,
                                      std::string delimiter, std::ostream &os) {
  for (size_t i = 0; i < shape.size(); i++) {
      if (i != 0)
        os << delimiter;
    os << shape[i];
  }
}

void CodeGenTileLangNPUIR::PrintType(DataType t,
                                     std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
      // ICHECK(t.si_scalar()) << "do not yet support vector types";
      // os << "void*";
      return;
  }

  if (t.is_void()) {
      //    os << "void";
      return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
        os << "f16";
      } else {
          fail = true;
      }
      break;
    case 32:
      os << "f32";
      break;
    case 64:
      os << "f64";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
        os << "bf16";
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t == DataType::Bool()) {
    os << "i1";
    return;
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "i1";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 4: {
      if (t.is_scalar()) {
          os << "i4";
          return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 8: {
      if (t.is_scalar()) {
          os << "i8";
          return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 16: {
      if (t.is_scalar()) {
          os << "i16";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
          os << "i32";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
          os << "i64";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail) {
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t;
}

void CodeGenTileLangNPUIR::PrintStorageScope(const std::string &scope,
                                             std::ostream &os) { // NOLINT(*)
}

void CodeGenTileLangNPUIR::VisitExpr_(const FloorDivNode *op,
                                      std::ostream &os) {
  // FIXME: The floor div in python is not the same as arith.divsi in negative
  // scenarios.
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "divsi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "divf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const FloorModNode *op,
                                      std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "remsi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "remf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const LTNode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi slt,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ult,", os, this);
  } else {
    PrintBinary(op, "cmpf olt,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const NENode *op, std::ostream &os) {
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ne,", os, this);
  } else {
    PrintBinary(op, "cmpf one,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const EQNode *op, std::ostream &os) {
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi eq,", os, this);
  } else {
    PrintBinary(op, "cmpf oeq,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const LENode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sle,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ule,", os, this);
  } else {
    PrintBinary(op, "cmpf ole,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const GENode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sge,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi uge,", os, this);
  } else {
    PrintBinary(op, "cmpf oge,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const GTNode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sgt,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ugt,", os, this);
  } else {
    PrintBinary(op, "cmpf ogt,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const CastNode *op, std::ostream &os) {
  bool srcIsFloat =
      op->value->dtype.is_float() || op->value->dtype.is_bfloat16();
  bool srcIsInt = op->value->dtype.is_int();
  bool srcIsUInt = op->value->dtype.is_uint();
  bool targetIsFloat = op->dtype.is_float() || op->dtype.is_bfloat16();
  bool targetIsInt = op->dtype.is_int();
  bool targetIsUInt = op->dtype.is_uint();
  auto val = PrintExpr(op->value);
  if (srcIsFloat && targetIsInt) {
    os << "arith.fptosi \%" << val << " : ";
  } else if (srcIsFloat && targetIsUInt) {
    os << "arith.fptoui \%" << val << " : ";
  } else if (srcIsInt && targetIsFloat) {
    os << "arith.sitofp \%" << val << " : ";
  } else if (srcIsUInt && targetIsFloat) {
    os << "arith.uitofp \%" << val << " : ";
  } else if (targetIsInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extsi \%" << val << " : ";
    } else {
      os << "arith.trunci \%" << val << " : ";
    }
  } else if (targetIsUInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extui \%" << val << " : ";
    } else {
      os << "arith.trunci \%" << val << " : ";
    }
  } else if (targetIsFloat) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extf \%" << val << " : ";
    } else {
      os << "arith.truncf \%" << val << " : ";
    }
  }
  PrintType(op->value->dtype, os);
  os << " to ";
  PrintType(op->dtype, os);
}

Array<String> CodeGenTileLangNPUIR::GenConvertIndex(Array<PrimExpr> exprs) {
  Array<String> cast_array;
  for (PrimExpr curr_expr : exprs) {
    if (auto curr_expr_int = curr_expr.as<IntImmNode>()) {
      cast_array.push_back(std::to_string(curr_expr_int->value));
    } else {
      DataType indice_type = curr_expr->dtype;
      std::ostringstream temp;
      std::string var_name = PrintExpr(curr_expr);
      if (!curr_expr.as<VarNode>()) {
        var_name = SSAGetID(var_name, indice_type);
      }
      temp << "arith.index_cast \%" << var_name << " : ";
      PrintType(indice_type, temp);
      temp << " to index";
      String cast_indice_name = "\%" + SSAGetID(temp.str(), indice_type);
      cast_array.push_back(cast_indice_name);
    }
  }
  return cast_array;
}

unsigned long ComputeOffset(Memref *src_buffer, Array<PrimExpr> op_offset) {
  if (src_buffer->var_offset)
    return -1;
  if (src_buffer->stride_int.size() != src_buffer->dim)
    return -1;
  unsigned long offset = src_buffer->offset;
  for (int i = 0; i < src_buffer->dim; i++) {
    const int64_t *op_off = as_const_int(op_offset[i]);
    if (op_off == nullptr)
      return -1;
    offset += (*op_off) * src_buffer->stride_int[i];
  }
  return offset;
}

String CodeGenTileLangNPUIR::GenSubviewFromRegion(const CallNode *region_node) {
  tvm::tl::RegionOp regionop(region_node->args, this->vmap);
  return GenSubviewFromRegion(regionop.GetBuffer(), regionop.GetRanges());
}

String CodeGenTileLangNPUIR::GenSubviewFromRegion(Buffer buffer_data,
                                                  Array<Range> range) {
  std::ostringstream temp;
  DataType buffer_type = buffer_data->dtype;
  String buffer_name = buffer_data->name;
  Array<PrimExpr> buffer_shape = buffer_data->shape;
  int dim = buffer_shape.size();
  Array<PrimExpr> region_shape, region_indeces;
  for (Range r : range) {
    region_shape.push_back(r.get()->extent);
    region_indeces.push_back(r.get()->min);
  }
  String buffer_name_val = "";
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[buffer_name])) {
    if (memrefInfo->is_arg) {
      buffer_name_val = buffer_name + "_Recast";
    } else {
      buffer_name_val = buffer_name;
    }
  } else {
    LOG(FATAL) << buffer_name << " should be a memref";
  }
  String new_buffer_name = buffer_name_val;
  String src_data_info = GetMemrefInfo(buffer_name_val);
  if (!(IsEqual(buffer_shape, region_shape) && AllZero(region_indeces))) {
    // get Indice name from BufferLoadNode, add type cast to index if needed.
    Array<String> cast_offset_array = GenConvertIndex(region_indeces);
    Array<String> cast_shape_array = GenConvertIndex(region_shape);
    // add new memref obj
    if (!dynamic_cast<Memref *>(type_info[buffer_name_val])) {
      LOG(FATAL) << buffer_name_val << " should be a memref";
    }
    unsigned long offset = ComputeOffset(
        dynamic_cast<Memref *>(type_info[buffer_name_val]), region_indeces);
    // gen subview
    new_buffer_name = buffer_name_val + "_subview";
    auto tempMemref = new Memref(
        new_buffer_name, region_shape, buffer_type,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->address_space,
        offset == -1,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->stride, offset);
    String dst_data_info = GetMemrefInfo(tempMemref);
    temp << "memref.subview \%" + buffer_name_val;
    // gen offset
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << cast_offset_array[i];
    }
    temp << "]";
    // gen shape
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << cast_shape_array[i];
    }
    temp << "]";
    // gen stride
    // Assuming the stride remains unchanged in region, set all 1.
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << "1";
    }
    temp << "]";
    temp << " : ";
    temp << src_data_info;
    temp << " to ";
    temp << dst_data_info;
    delete tempMemref;
    new_buffer_name = SSAGetID(temp.str(), buffer_type);
    this->type_info[new_buffer_name] = new Memref(
        new_buffer_name, region_shape, buffer_type,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->address_space,
        offset == -1,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->stride, offset);
  }
  return new_buffer_name;
}

/// Generate hivm.hir.load or hivm.hir.store for tl.ascend_copy.
/// before:
///   T.ascend_copy(T.region(A[bx, by], 1, 128, 256), T.region(A_VEC[0, 0],
///   2, 128, 256))
/// after:
///   memref.reinterpret_cast; memref.subview; memref.subview; hivm.hir.store/load
void CodeGenTileLangNPUIR::AscendCopyCodegen(const CallNode* op, std::ostream& os){
  tvm::tl::AscendCopy npuirop(op->args, this->vmap);
  // gen memref.subview
  String src_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  String dst_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  // gen hivm.ir.load / hivm.ir.store
  this->PrintIndent();
  if (!dynamic_cast<Memref *>(type_info[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a memref";
  }
  if (!dynamic_cast<Memref *>(type_info[dst_data_name])) {
    LOG(FATAL) << dst_data_name << " should be a memref";
  }
  if (dynamic_cast<Memref *>(type_info[src_data_name])->address_space == "gm") {
    this->stream << "hivm.hir.load";
  } else if (dynamic_cast<Memref *>(type_info[dst_data_name])->address_space ==
             "gm") {
    this->stream << "hivm.hir.store";
  } else if (dynamic_cast<Memref *>(type_info[src_data_name])->address_space ==
                 "ub" &&
             dynamic_cast<Memref *>(type_info[dst_data_name])->address_space ==
                 "ub") {
    this->stream << "hivm.hir.copy";
  }
  this->stream << " ins(" << "\%" << src_data_name << " : "
               << GetMemrefInfo(src_data_name) << ")";
  this->stream << " outs(" << "\%" << dst_data_name << " : "
               << GetMemrefInfo(dst_data_name) << ")";
  // TODO: error: custom op 'init_out_buffer' is unknown
  // this->stream << " left_padding_num = %c0 : index init_out_buffer =
  // false";
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::VexpCodegen(const CallNode* op, std::ostream& os){
  // Generate hivm.hir.vexp for tl.npuir_exp.
  // before:
  //   T.npuir_exp(T.region(A_VEC[0], 1, 1024), T.region(B_VEC[0], 2, 1024))
  // after:
  //   hivm.hir.vexp ins(%9 : memref<1024xf32, #hivm.address_space<ub>>)
  //       outs(%16 : memref<1024xf32, #hivm.address_space<ub>>)
  tvm::tl::NpuirExp npuirop(op->args, this->vmap);
  std::string in_data_name = "", out_data_name = "";
  Array<PrimExpr> buffer_shape0 = npuirop.src->shape,
                  buffer_shape1 = npuirop.dst->shape;
  in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  this->PrintIndent();
  this->stream << "hivm.hir.vexp ins(\%" << in_data_name << " : "
               << this->type_info[in_data_name]->printType() << ") outs(\%"
               << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  auto dims = broadcastAttrCodegen(buffer_shape0, buffer_shape1);
  if (dims != "") {
    this->stream << " broadcast" << dims;
  }
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::VbrcCodegen(const CallNode *op, std::ostream &os) {
  // Generate hivm.hir.vbrc for tl.npuir_vbrc.
  tvm::tl::NpuirBrc npuirop(op->args, this->vmap);
  std::string in_data_name = "", out_data_name = "";
  DataType outDtype = npuirop.dst->dtype;
  Array<PrimExpr> buffer_shape0, buffer_shape1;
  if (auto intImm = npuirop.in.as<IntImm>()) {
    auto immObj = intImm.value();
    in_data_name = PrintExpr(immObj);
    if (intImm.value()->dtype != outDtype) {
      in_data_name = ScalarConvertType(&immObj, outDtype);
    }
    std::ostringstream temp;
    PrintType(outDtype, temp);
    this->type_info[in_data_name] = new Scalar(in_data_name, temp.str());
  } else if (auto floatImm = npuirop.in.as<FloatImm>()) {
    auto immObj = floatImm.value();
    in_data_name = PrintExpr(immObj);
    if (floatImm.value()->dtype != outDtype) {
      in_data_name = ScalarConvertType(&immObj, outDtype);
    }
    std::ostringstream temp;
    PrintType(outDtype, temp);
    this->type_info[in_data_name] = new Scalar(in_data_name, temp.str());
  } else {
    buffer_shape0 = npuirop.src->shape;
    in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  }
  buffer_shape1 = npuirop.dst->shape;
  out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  this->PrintIndent();
  this->stream << "hivm.hir.vbrc ins(\%" << in_data_name << " : "
               << this->type_info[in_data_name]->printType() << ") outs(\%"
               << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  auto dims = broadcastAttrCodegen(buffer_shape0, buffer_shape1);
  if (dims != "") {
    this->stream << " broadcast_dims" << dims;
  }
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::VcastCodegen(const CallNode *op, std::ostream &os) {
  // Generate hivm.hir.vcast for tl.npuir_cast.
  tvm::tl::NpuirCast npuirop(op->args, this->vmap);
  std::string in_data_name = "", out_data_name = "";
  Array<PrimExpr> buffer_shape0 = npuirop.src->shape,
                  buffer_shape1 = npuirop.dst->shape;
  in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  std::string round_mode = op->args[2].as<StringImmNode>()->value;
  this->PrintIndent();
  this->stream << "hivm.hir.vcast ins(\%" << in_data_name << " : "
               << this->type_info[in_data_name]->printType() << ") outs(\%"
               << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  this->stream << " round_mode = <" << round_mode << ">";
  auto dims = broadcastAttrCodegen(buffer_shape0, buffer_shape1);
  if (dims != "") {
    this->stream << " broadcast" << dims;
  }
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::VreduceCodegen(const CallNode *op,
                                          std::ostream &os) {
  // Generate hivm.hir.vreduce for tl.npuir_reduce.
  tvm::tl::NpuirReduce npuirop(op->args, this->vmap);
  std::string in_data_name = "", out_data_name = "";
  Array<PrimExpr> buffer_shape0 = npuirop.src->shape,
                  buffer_shape1 = npuirop.dst->shape;
  in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  std::vector<int> reduceDims;
  reduceDims = npuirop.reduce_dims;
  std::string reduce_mode = npuirop.reduce_mode;
  this->PrintIndent();
  this->stream << "hivm.hir.vreduce <" << reduce_mode << "> ins(\%"
               << in_data_name << " : "
               << this->type_info[in_data_name]->printType() << ") outs(\%"
               << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  this->stream << " reduce_dims = [";
  for (auto dim : reduceDims) {
    this->stream << dim;
    if (dim != reduceDims.back()) {
      this->stream << ", ";
    }
  }
  this->stream << "]";
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::Nd2NzCodegen(const CallNode* op, std::ostream& os){
  // Generate hivm.hir.nd2nz for tl.npuir_load_nd2nz.
  tvm::tl::NpuirNd2nz npuirop(op->args, this->vmap);
  // gen memref.subview
  String src_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  String dst_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  bool dst_continuous = npuirop.dst_continuous;

  // gen hivm.hir.nd2nz
  this->PrintIndent();
  this->stream << "hivm.hir.nd2nz";
  if (dst_continuous)
    this->stream << " {dst_continuous}";
  this->stream << " ins(%" << src_data_name << " : "
               << GetMemrefInfo(src_data_name) << ")";
  this->stream << " outs(%" << dst_data_name << " : "
               << GetMemrefInfo(dst_data_name) << ")";
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::FixpipeCodegen(const CallNode* op, std::ostream& os){
  // Generate hivm.hir.fixpipe for tl.npuir_store_fixpipe.
  tvm::tl::NpuirFixpipe npuirop(op->args, this->vmap);
  // gen memref.subview
  String src_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  String dst_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  bool enable_nz2nd = npuirop.enable_nz2nd;
  bool channel_split = npuirop.channel_split;
  int pre_relu_mode = npuirop.pre_relu_mode;

  // gen hivm.hir.fixpipe
  this->PrintIndent();

  auto src_dtype = npuirop.src->dtype;
  auto dst_dtype = npuirop.dst->dtype;
  std::string pre_quant_attr = "";
  if (src_dtype != dst_dtype) {
    if (src_dtype == DataType::Float(32) && dst_dtype == DataType::Float(16)) {
      pre_quant_attr = "F322F16";
    } else if (src_dtype == DataType::Float(32) && dst_dtype == DataType::BFloat(16)) {
      pre_quant_attr = "F322BF16";
    } else if (src_dtype == DataType::Int(32) && dst_dtype == DataType::Int(8)) {
      pre_quant_attr = "S322I8";
    } else {
      assert(false && "Unexpected pre-quant mode. Should not reach here.");
    }
  }

  this->stream << "hivm.hir.fixpipe {";
  this->stream << "channel_split = " << (channel_split ? "true" : "false");
  this->stream << (enable_nz2nd ? ", enable_nz2nd" : "");
  this->stream << ", pre_relu = #hivm.fixpipe_pre_relu_mode<"
               << fixpipe_pre_relu_mode[pre_relu_mode] << ">";
  if (pre_quant_attr != "") {
    this->stream << ", pre_quant = #hivm.fixpipe_pre_quant_mode<" << pre_quant_attr << ">";
  }
  this->stream << "}";
  this->stream << " ins(%" << src_data_name << " : "
               << GetMemrefInfo(src_data_name) << ")";
  this->stream << " outs(%" << dst_data_name << " : "
               << GetMemrefInfo(dst_data_name) << ")";
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::BarrierCodegen(const CallNode* op, std::ostream& os) {
  tvm::tl::NpuirPipeBarrier npuirop (op->args, this->vmap);
  this->PrintIndent();
  this->stream << "hivm.hir.pipe_barrier[<";
  this->stream << npuirop.pipe_type;
  this->stream << ">]\n";
}

std::string CodeGenTileLangNPUIR::PrintID(PrimExpr id) {
  auto raw_type = id.dtype();
  auto id_name = SSAGetID(PrintExpr(id), raw_type);
  assert(raw_type.is_int() || raw_type.is_uint());
  if (raw_type.bits() < FLAG_ID_BITS) {
    std::ostringstream temp;
    temp << "arith.";
    if (raw_type.is_int()) {
      temp << "extsi %";
    } else {
      temp << "extui %";
    }
    temp << id_name << " : ";
    PrintType(raw_type, temp);
    temp << " to i64\n";
    id_name = SSAGetID(temp.str(), DataType::Int(FLAG_ID_BITS));
  }
  return "%" + id_name;
}

template <typename T>
void CodeGenTileLangNPUIR::PipeFlagCodegen(const T &sync_op, std::ostream &os) {
  std::string event_id;
  if (auto *int_imm = sync_op.event_id.template as<tvm::tir::IntImmNode>()) {
    event_id = "<EVENT_ID" + std::to_string(int_imm->value) + ">";
  } else {
    event_id = PrintID(sync_op.event_id);
  }

  this->PrintIndent();
  this->stream << "hivm.hir.";
  this->stream << T::op;
  this->stream << "_flag[<" << sync_op.pipe1 << ">, <" << sync_op.pipe2 << ">, "
               << event_id << "]\n";
}

template <typename T>
void CodeGenTileLangNPUIR::SyncBlockWaitCodegen(const T &sync_op, std::ostream& os) {
  std::string flag_id;
  if (auto *int_imm = sync_op.flag_id.template as<tvm::itr::IntImmNode()) {
    flag_id = std::to_string(int_imm->value);
  } else {
    flag_id = PrintID(sync_op.flag_id);
  }
  this->PrintIndent();
  this->stream << "hivm.hir.sync_block_wait[<";
  this->stream << coretype_syncblock_map[current_coretype];
  this->stream << ">, <PIPE_S>, <" << sync_op.pipe_type
               << ">] flag = " << flag_id << "\n";
}

template <typename T>
void CodeGenTileLangNPUIR::SyncBlockSetCodegen(const T &sync_op, std::ostream& os){
  std::string flag_id;
  if (auto *int_imm = sync_op.flag_id.template as<tvm::tir::IntImmNode>()) {
    flag_id = std::to_string(int_imm->value);
  } else {
    flag_id = PrintID(sync_op.flag_id);
  }
  this->PrintIndent();
  this->stream << "hivm.hir.sync_block_set[<";
  this->stream << coretype_syncblock_map[current_coretype] << ">, <";
  this->stream << sync_op.pipe_type;
  this->stream << ">, <PIPE_S>] flag = " << flag_id;
  this->stream << " syn_instr_mode = <" << SyncBlockMode_str[sync_op.mode];
  this->stream << ">\n";
}

void CodeGenTileLangNPUIR::DotCodegen(const CallNode* op, std::ostream& os){
  // Generate hivm.hir.mmadL1 for tl.npuir_dot.
  // before:
  //   T.npuir_dot(T.region(A_BUF[0, 0], 1, 128, 1024),
  //               T.region(B_BUF[0, 0], 1, 1024, 256),
  //               T.region(C_BUF[0, 0], 3, 128, 256), T.bool(True))
  // after:
  // hivm.hir.mmadL1 ins(%alloc_8,  %alloc_5,  %true,  %c128,  %c64,  %c64 :
  //                     memref<128x64xf16,  #hivm.address_space<cbuf>>,
  //                     memref<64x64xf16,  #hivm.address_space<cbuf>>,
  //                     i1,  index,  index,  index)
  //                 outs(%alloc_9 : memref<128x64xf32,
  //                      #hivm.address_space<cc>>)
  tvm::tl::NpuirDot npuirop(op->args, this->vmap);
  Buffer a_buffer = npuirop.src0;
  Buffer b_buffer = npuirop.src1;
  Buffer c_buffer = npuirop.dst;
  Array<PrimExpr> a_region_shape, b_region_shape;
  for (int i = 0; i < npuirop.src0_range.size(); i++) {
    a_region_shape.push_back(npuirop.src0_range[i].get()->extent);
    b_region_shape.push_back(npuirop.src1_range[i].get()->extent);
  }
  auto init_c_name = SSAGetID(PrintExpr(npuirop.initC), op->args[3]->dtype);

  auto GetRealName = [this](const PrimExpr &extent) {
    std::ostringstream temp;
    auto real_name = SSAGetID(PrintExpr(extent), extent.dtype());
    temp << "arith.index_cast \%" << real_name << " : ";
    PrintType(extent.dtype(), temp);
    temp << " to index";
    real_name = SSAGetID(temp.str(), extent.dtype());
    return real_name;
  };
  auto real_m_name = GetRealName(a_region_shape[0]);
  auto real_k_name = GetRealName(b_region_shape[0]);
  auto real_n_name = GetRealName(b_region_shape[1]);

  this->PrintIndent();
  this->stream << "hivm.hir.mmadL1";
  if (npuirop.a_transpose || npuirop.b_transpose) {
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
  this->stream << ", i1, index, index, index";
  this->stream << ")";
  this->stream << " outs(%" << type_info[c_buffer->name]->var_id;
  this->stream << " : " << GetMemrefInfo(c_buffer->name);
  this->stream << ")\n";
}

} // namespace codegen
} // namespace tvm