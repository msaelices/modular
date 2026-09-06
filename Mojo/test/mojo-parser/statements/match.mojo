# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# A function we can call with minimal IR gruff but still verify the right
# code is put out in the right place.
def case_callee[p: Int](): pass


# CHECK-LABEL: lit.fn @"match_same_indent
# CHECK-NEXT:    hlcf.elif {
# CHECK-NEXT:      [[L0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:      [[EQ0:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L0]])
# CHECK-NEXT:      [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ0]])
# CHECK-NEXT:      hlcf.elif.yield [[B0]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } {
# CHECK-NEXT:      [[L1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
# CHECK-NEXT:      [[EQ1:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L1]])
# CHECK-NEXT:      [[B1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ1]])
# CHECK-NEXT:      hlcf.elif.yield [[B1]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } {
# CHECK-NEXT:      [[T2:%.*]] = kgen.param.constant: scalar<bool> = <true>
# CHECK-NEXT:      hlcf.elif.yield [[T2]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
def match_same_indent(x: Int):
    __match x:
    case 0:
        case_callee[0]()
    case 1:
        case_callee[1]()
    case _:
        case_callee[2]()


# CHECK-LABEL: lit.fn @"match_indented_cases
# CHECK:       hlcf.elif {
# CHECK:         kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } {
# CHECK:         kgen.param.constant: scalar<bool> = <true>
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.yield
# CHECK:       }
def match_indented_cases(x: Int):
    __match x:
        case 0:
            case_callee[0]()
        case _:
            case_callee[1]()


# CHECK-LABEL: lit.fn @"match_tuple_subject
# CHECK:       hlcf.elif {
# CHECK:         kgen.param.constant: scalar<bool> = <false>
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } {
# CHECK:         kgen.param.constant: scalar<bool> = <true>
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.yield
# CHECK:       }
def match_tuple_subject(point: Tuple[Int, Int]):
    __match point:
    case (0, 0):
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_with_guard
# CHECK-NEXT:    hlcf.elif {
# CHECK-NEXT:      [[Z0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:      [[NE0:%.*]] = lit.call {{.*}}@"__ne__({{.*}}(%c, [[Z0]])
# CHECK-NEXT:      [[B0:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[NE0]])
# CHECK-NEXT:      hlcf.elif.yield [[B0]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } {
# CHECK-NEXT:      [[L1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:      [[EQ1:%.*]] = lit.call {{.*}}@"__eq__({{.*}}(%x, [[L1]])
# CHECK-NEXT:      [[B1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[EQ1]])
# CHECK-NEXT:      [[G1:%.*]] = hlcf.elif -> !kgen.scalar<bool> {
# CHECK-NEXT:        hlcf.elif.yield [[B1]]
# CHECK-NEXT:      } then {
# CHECK-NEXT:        [[Z1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK-NEXT:        [[NE1:%.*]] = lit.call {{.*}}@"__ne__({{.*}}(%c, [[Z1]])
# CHECK-NEXT:        [[GB1:%.*]] = lit.call {{.*}}@"__mlir_bool__(::Bool)"([[NE1]])
# CHECK-NEXT:        hlcf.yield [[GB1]] : !kgen.scalar<bool>
# CHECK-NEXT:      } else {
# CHECK-NEXT:        [[F1:%.*]] = kgen.param.constant: scalar<bool> = <false>
# CHECK-NEXT:        hlcf.yield [[F1]] : !kgen.scalar<bool>
# CHECK-NEXT:      }
# CHECK-NEXT:      hlcf.elif.yield [[G1]]
# CHECK-NEXT:    } then {
# CHECK-NEXT:      lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    } else {
# CHECK-NEXT:      hlcf.yield
# CHECK-NEXT:    }
def match_with_guard(x: Int, c: Int):
    __match x:
    case _ if c != 0:
        case_callee[0]()
    case 0 if c != 0:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_case_body
# CHECK:       hlcf.elif {
# CHECK:         kgen.param.constant: !Int = <{:scalar<index> 0}>
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         %inside_case = lit.var.decl "inside_case"
# CHECK:         hlcf.yield
# CHECK:       } {
# CHECK:         kgen.param.constant: scalar<bool> = <true>
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.yield
# CHECK:       }
def match_case_body(x: Int):
    __match x:
    case 0:
        var inside_case: Int
    case _:
        case_callee[0]()


# CHECK-LABEL: lit.fn @"match_float
# CHECK:       hlcf.elif {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
def match_float(x: Float64):
    __match x:
    case 0.0:
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_string
# CHECK:       hlcf.elif {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
def match_string(x: String):
    __match x:
    case "a":
        case_callee[0]()
    case _:
        case_callee[1]()


# CHECK-LABEL: lit.fn @"match_bool
# CHECK:       hlcf.elif {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       } {
def match_bool(x: Bool):
    __match x:
    case True:
        case_callee[0]()
    case False:
        case_callee[1]()
    case _:
        case_callee[2]()


# Enum-like Color with inferred-member patterns (e.g. `case .red`).
struct Color(ImplicitlyCopyable):
    comptime red = Color()
    comptime green = Color()
    comptime blue = Color()

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True


# CHECK-LABEL: lit.fn @"match_color
# CHECK:       hlcf.elif {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 0
# CHECK:         hlcf.yield
# CHECK:       } {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 1
# CHECK:         hlcf.yield
# CHECK:       } {
# CHECK:         lit.call {{.*}}@"__eq__(
# CHECK:         lit.call {{.*}}@"__mlir_bool__(::Bool)"
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 2
# CHECK:         hlcf.yield
# CHECK:       } {
# CHECK:         kgen.param.constant: scalar<bool> = <true>
# CHECK:         hlcf.elif.yield
# CHECK:       } then {
# CHECK:         lit.call {{.*}}@"case_callee{{.*}}<index> 3
# CHECK:         hlcf.yield
# CHECK:       } else {
# CHECK:         hlcf.yield
# CHECK:       }
def match_color(c: Color):
    __match c:
    case Color.red:  # explicit enum case.
        case_callee[0]()
    case .green:     # inferred case.
        case_callee[1]()
    case .blue:     # inferred case.
        case_callee[2]()
    case _:
        case_callee[3]()
