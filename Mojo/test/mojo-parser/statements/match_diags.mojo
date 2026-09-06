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


def match_bare_identifier(x: Int, y: Int):
    __match x:
    # Bare names are reserved for a future implicit-binding syntax; they are
    # not "match against the existing value named y".
    # expected-error @below {{bare identifier 'y' is not a valid match pattern; use 'var y' or 'ref y' to bind a name}}
    case y:
        pass
    case _:
        pass


# expected-error @+1 {{'__match' must be contained in a function}}
__match 1:
    case 0:
        pass

def various_match_issues(a: Int, point: Tuple[Int, Int], value: String):
    __match a:
    case 0:
        var x = 42
    case _:
        # expected-error @+1 {{use of unknown declaration 'x'}}
        _ = x

    __match point:
    # expected-error @+1 {{cannot match value of 'Tuple[Int, Int]' of 2 elements against a pattern with 3 elements}}
    case (0, 0, 0):
        pass
    case _:
        pass

    __match point:
    # `var` takes a starred list, so write the collision as `var (x, x)`.
    # expected-error @+2 {{invalid redefinition of 'x'}}
    # expected-note @+1 {{previous definition here}}
    case var (x, x):
        pass
    case _:
        pass

    # Nested var/ref patterns should warn.
    __match value:
    # expected-warning @+1 {{nested 'var' or 'ref' patterns are redundant, remove the outer pattern}}
    case var ref x:
        _ = x.byte_length()
    # expected-warning @+1 {{nested 'var' or 'ref' patterns are redundant, remove the outer pattern}}
    case ref var y:
        _ = y.byte_length()
    case _:
        pass

    # Use of a name without var/ref should be an error (for now).
    __match value:
    # expected-error @+1 {{bare identifier 'x' is not a valid match pattern; use 'var x' or 'ref x' to bind a name}}
    case x:
        pass
    case _:
        pass

    __match a:
    # expected-error @+1 {{expected a name after 'as'}}
    case 0 as 1:
        pass
    case _:
        pass

    __match point:
    # expected-error @+1 {{expected a name after 'as'}}
    case (0 as 1, 2):
        pass
    case _:
        pass
