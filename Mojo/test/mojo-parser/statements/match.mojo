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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# Basic recognition of `__match` / `case`. IR lowering is not implemented yet,
# so well-formed matches diagnose as unimplemented after parsing.

def match_same_indent(x: Int):
    # expected-error @+1 {{'__match' statement is not implemented yet}}
    __match x:
    case 0:
        pass
    case 1:
        pass
    case _:
        pass


def match_indented_cases(x: Int):
    # expected-error @+1 {{'__match' statement is not implemented yet}}
    __match x:
        case 0:
            pass
        case _:
            pass


def match_tuple_subject(point: Tuple[Int, Int]):
    # expected-error @+1 {{'__match' statement is not implemented yet}}
    __match point:
    case (0, 0):
        pass
    case _:
        pass


def match_missing_cases(x: Int):
    # expected-error @+1 {{'__match' statement must have at least one 'case' block}}
    __match x:


# expected-error @+1 {{'__match' must be contained in a function}}
__match 1:
    case 0:
        pass
