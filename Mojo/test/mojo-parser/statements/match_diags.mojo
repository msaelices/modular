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

def match_bad_subject_skips_same_indent_cases():
    # expected-error @+1 {{use of unknown declaration 'no_such_subject'}}
    __match no_such_subject:
    case 0:
        pass
    case 1 if True:
        pass
    # Recovery must resume here, not leave a dangling 'case' for the outer suite.
    # expected-error @+1 {{use of unknown declaration 'also_missing'}}
    _ = also_missing


def match_bad_subject_skips_indented_cases():
    # expected-error @+1 {{use of unknown declaration 'no_such_subject'}}
    __match no_such_subject:
        case 0:
            pass
        case _:
            pass
    # expected-error @+1 {{use of unknown declaration 'also_missing'}}
    _ = also_missing


def match_missing_cases(x: Int):
    # expected-error @+1 {{'__match' statement must have at least one 'case' block}}
    __match x:

    __match x:
    case foo(): # expected-error {{expression is not a valid match pattern}}
        pass


# expected-error @+1 {{'__match' must be contained in a function}}
__match 1:
    case 0:
        pass

def match_scoping(a: Int):
    __match a:
    case 0:
        var x = 42
    case _:
        # expected-error @+1 {{use of unknown declaration 'x'}}
        _ = x


def match_tuple_arity_mismatch(point: Tuple[Int, Int, Int]):
    __match point:
    # expected-error @+1 {{cannot match value of 'Tuple[Int, Int, Int]' of 3 elements against a pattern with 2 elements}}
    case (0, 0):
        pass
    case _:
        pass
