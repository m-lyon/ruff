//! A small public facade for tools that embed ty and need to type-check calls they build
//! themselves, without Python source for the call site.
//!
//! The facade returns plain data and only names public types, so callers do not depend on ty's
//! internal call-binding structures.

use ruff_python_ast::name::Name;
use ty_module_resolver::{ModuleName, resolve_module_confident};
use ty_python_core::definition::{Definition, DefinitionState};
use ty_python_core::scope::ScopeId;
use ty_python_core::{ProgramFile, place_table, use_def_map};

use crate::Db;
use crate::place::imported_symbol;
use crate::reachability::evaluate_reachability;
use crate::types::call::bind::{BindingError, CallableBinding};
use crate::types::call::{Argument, CallArguments, CallError};
use crate::types::{KnownClass, ProgramEnvironment, Type};

/// The type of `path` looked up on `module`: a module-level symbol, then attributes of it.
///
/// For example, `module_member(db, env, "pkg.models", &["Model", "create"])` is the type of
/// `pkg.models.Model.create`. Re-exports are followed as for an `import` statement. Returns
/// `None` if the module does not resolve or a name in `path` is not defined.
pub fn module_member<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    module: &ModuleName,
    path: &[&str],
) -> Option<Type<'db>> {
    let (first, rest) = path.split_first()?;
    let module = resolve_module_confident(db, env.resolver_environment(db), module)?;
    let file = module
        .file(db)
        .map(|file| ProgramFile::new(db, file, env.program(db)));
    let mut ty = imported_symbol(db, env, file, first, None)
        .place
        .ignore_possibly_undefined()?;
    for name in rest {
        ty = ty.member(db, env, name).place.ignore_possibly_undefined()?;
    }
    Some(ty)
}

/// The type `Literal["value"]`.
pub fn str_literal<'db>(db: &'db dyn Db, value: &str) -> Type<'db> {
    Type::string_literal(db, value)
}

/// The type `Literal[value]`.
pub fn int_literal<'db>(value: i64) -> Type<'db> {
    Type::int_literal(value)
}

/// The type of an instance of `float` itself. A `float` annotation means `int | float`.
pub fn float_instance<'db>(db: &'db dyn Db, env: &ProgramEnvironment<'db>) -> Type<'db> {
    KnownClass::Float.to_instance(db, env)
}

/// The type `str`.
pub fn str_instance<'db>(db: &'db dyn Db, env: &ProgramEnvironment<'db>) -> Type<'db> {
    KnownClass::Str.to_instance(db, env)
}

/// The type `list[Unknown]`.
pub fn list_of_unknown<'db>(db: &'db dyn Db, env: &ProgramEnvironment<'db>) -> Type<'db> {
    KnownClass::List.to_specialized_instance(db, env, &[Type::unknown()])
}

/// The type `dict[str, Unknown]`.
pub fn dict_of_str_unknown<'db>(db: &'db dyn Db, env: &ProgramEnvironment<'db>) -> Type<'db> {
    KnownClass::Dict.to_specialized_instance(db, env, &[str_instance(db, env), Type::unknown()])
}

/// The type of an instance of `class`, such as `DictConfig` for the class object `DictConfig`.
/// `None` if `class` is not a class.
pub fn instance_of<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    class: Type<'db>,
) -> Option<Type<'db>> {
    class.to_instance_approximation(db, env)
}

/// How an argument is passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg<'a> {
    Positional,
    Keyword(&'a str),
}

/// The result of [`check_call`].
#[derive(Clone, Debug)]
pub struct CallCheck<'db> {
    /// The call's return type, or the best approximation of it if the call has errors.
    pub return_type: Type<'db>,
    /// Arguments whose type is not assignable to their parameter's type.
    pub errors: Vec<ArgumentTypeError<'db>>,
    /// The parameter each argument was matched to, in the overload that was selected. A
    /// constructor call can match an argument both in `__new__` and in `__init__`.
    pub matched_parameters: Vec<MatchedParameter<'db>>,
    /// Whether an overloaded callable had no overload that matched the arguments.
    pub no_matching_overload: bool,
}

/// An argument matched to a parameter.
#[derive(Clone, Copy, Debug)]
pub struct MatchedParameter<'db> {
    /// The index of the argument in the `args` passed to [`check_call`].
    pub argument_index: usize,
    /// The parameter's annotated type, or `Unknown` if it has no annotation.
    pub annotated_type: Type<'db>,
}

/// An argument whose type is not assignable to the type of the parameter it was matched to.
#[derive(Clone, Debug)]
pub struct ArgumentTypeError<'db> {
    /// The index of the argument in the `args` passed to [`check_call`].
    pub argument_index: usize,
    /// The parameter's name, if it has one.
    pub parameter: Option<String>,
    pub expected: Type<'db>,
    pub found: Type<'db>,
}

/// Binds `args` to `callable`'s parameters and checks their types, as ty checks the call
/// `callable(*args)`.
///
/// Only argument type errors are reported. Other binding errors, such as a missing or unknown
/// argument, are not. For an overloaded callable that no overload matches, only
/// [`CallCheck::no_matching_overload`] is set.
pub fn check_call<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    callable: Type<'db>,
    args: &[(Arg<'_>, Type<'db>)],
) -> CallCheck<'db> {
    let arguments: CallArguments<'_, 'db> = args
        .iter()
        .map(|(arg, ty)| {
            let arg = match arg {
                Arg::Positional => Argument::Positional,
                Arg::Keyword(name) => Argument::Keyword(name),
            };
            (arg, Some(*ty))
        })
        .collect();
    let bindings = callable
        .try_call(db, env, &arguments)
        .unwrap_or_else(|CallError(_, bindings)| *bindings);

    let mut check = CallCheck {
        return_type: bindings.return_type(db, env),
        errors: Vec::new(),
        matched_parameters: Vec::new(),
        no_matching_overload: false,
    };
    bindings.visit_type_context_callables(&mut |binding| collect_binding(binding, &mut check));
    check
}

fn collect_binding<'db>(binding: &CallableBinding<'db>, check: &mut CallCheck<'db>) {
    let overload = match binding.matching_overloads().next() {
        Some((_, overload)) => overload,
        None => match binding.overloads() {
            [overload] => overload,
            [] => return,
            _ => {
                check.no_matching_overload = true;
                return;
            }
        },
    };

    // A bound method's receiver is matched as a synthetic first argument.
    let matches = overload
        .argument_matches()
        .get(usize::from(binding.bound_type.is_some())..)
        .unwrap_or_default();
    let parameters = overload.signature.parameters();
    let parameter_name = |argument_index: usize| match matches.get(argument_index)?.parameters[..] {
        [parameter] => parameters[parameter.index].name().map(Name::to_string),
        _ => None,
    };

    for (argument_index, matched) in matches.iter().enumerate() {
        for parameter in &matched.parameters {
            check.matched_parameters.push(MatchedParameter {
                argument_index,
                annotated_type: parameters[parameter.index].annotated_type(),
            });
        }
    }

    for error in overload.errors() {
        if let BindingError::InvalidArgumentType {
            argument_index: Some(argument_index),
            expected_ty,
            provided_ty,
            ..
        } = error
        {
            check.errors.push(ArgumentTypeError {
                argument_index: *argument_index,
                parameter: parameter_name(*argument_index),
                expected: *expected_ty,
                found: *provided_ty,
            });
        }
    }
}

/// Definitions of `name` in `scope` that may be reachable for the program's Python version and
/// platform: its bindings and declarations, in source order.
///
/// A definition whose reachability ty cannot decide is kept. Imports are not followed.
pub fn reachable_definitions<'db>(
    db: &'db dyn Db,
    scope: ScopeId<'db>,
    name: &str,
) -> Vec<Definition<'db>> {
    let Some(symbol) = place_table(db, scope).symbol_id(name) else {
        return Vec::new();
    };
    let use_def = use_def_map(db, scope);
    let bindings = use_def.reachable_symbol_bindings(symbol).map(|binding| {
        (
            binding.binding_order,
            binding.binding,
            binding.reachability_constraint,
        )
    });
    let declarations = use_def
        .reachable_symbol_declarations(symbol)
        .map(|declaration| {
            (
                declaration.declaration_order,
                declaration.declaration,
                declaration.reachability_constraint,
            )
        });

    // Bindings and declarations share one ordering within the scope, which follows the source.
    // A definition that both binds and declares, such as a `def`, appears in both.
    let mut entries: Vec<_> = bindings
        .chain(declarations)
        .filter_map(|(order, state, constraint)| match state {
            DefinitionState::Defined(definition)
                if evaluate_reachability(db, &*use_def, constraint).may_be_true() =>
            {
                Some((order, definition))
            }
            _ => None,
        })
        .collect();
    entries.sort_by_key(|(order, _)| *order);

    let mut definitions = Vec::with_capacity(entries.len());
    for (_, definition) in entries {
        if !definitions.contains(&definition) {
            definitions.push(definition);
        }
    }
    definitions
}

#[cfg(test)]
mod tests {
    use ruff_db::files::system_path_to_file;
    use ruff_db::parsed::parsed_module;
    use ruff_python_ast::PythonVersion;
    use ty_module_resolver::ModuleName;
    use ty_python_core::definition::DefinitionKind;
    use ty_python_core::scope::NodeWithScopeRef;
    use ty_python_core::{global_scope, semantic_index};

    use super::*;
    use crate::db::tests::{TestDb, TestDbBuilder};

    fn db_with(files: &[(&str, &str)]) -> TestDb {
        files
            .iter()
            .fold(TestDbBuilder::new(), |builder, (path, content)| {
                builder.with_file(path, content)
            })
            .build()
            .expect("valid TestDb setup")
    }

    fn member<'db>(db: &'db TestDb, module: &str, path: &[&str]) -> Option<Type<'db>> {
        let module = ModuleName::new(module).expect("valid module name");
        module_member(db, &db.program_environment(), &module, path)
    }

    fn display(db: &TestDb, ty: Type<'_>) -> String {
        ty.display(db, &db.program_environment()).to_string()
    }

    /// `check_call` on `module.path`, with its errors rendered as
    /// `argument index: parameter: expected <- found`.
    fn call(
        db: &TestDb,
        module: &str,
        path: &[&str],
        args: &[(Arg<'_>, Type<'_>)],
    ) -> (String, Vec<String>, bool) {
        let callable = member(db, module, path).expect("callable resolves");
        let check = check_call(db, &db.program_environment(), callable, args);
        let errors = check
            .errors
            .iter()
            .map(|error| {
                format!(
                    "{}: {}: {} <- {}",
                    error.argument_index,
                    error.parameter.as_deref().unwrap_or("?"),
                    display(db, error.expected),
                    display(db, error.found),
                )
            })
            .collect();
        (
            display(db, check.return_type),
            errors,
            check.no_matching_overload,
        )
    }

    #[test]
    fn module_member_finds_classes_functions_and_methods() {
        let db = db_with(&[
            (
                "/src/pkg/models.py",
                "class Model:\n    def create(self) -> int: ...\n    @classmethod\n    def build(cls) -> 'Model': ...\ndef make() -> Model: ...\n",
            ),
            (
                "/src/pkg/__init__.py",
                "from .models import Model as Model\n",
            ),
        ]);

        assert_eq!(
            display(&db, member(&db, "pkg.models", &["Model"]).unwrap()),
            "<class 'Model'>"
        );
        assert_eq!(
            display(&db, member(&db, "pkg.models", &["make"]).unwrap()),
            "def make() -> Model"
        );
        assert_eq!(
            display(
                &db,
                member(&db, "pkg.models", &["Model", "create"]).unwrap()
            ),
            "def create(self) -> int"
        );
        assert_eq!(
            display(&db, member(&db, "pkg.models", &["Model", "build"]).unwrap()),
            "bound method <class 'Model'>.build() -> Model"
        );
        assert_eq!(
            display(&db, member(&db, "pkg", &["Model"]).unwrap()),
            "<class 'Model'>"
        );
        assert_eq!(
            display(&db, member(&db, "collections", &["OrderedDict"]).unwrap()),
            "<class 'OrderedDict'>"
        );
        assert!(member(&db, "pkg.models", &["Missing"]).is_none());
        assert!(member(&db, "pkg.models", &["Model", "missing"]).is_none());
        assert!(member(&db, "missing", &["Model"]).is_none());
        assert!(member(&db, "pkg", &[]).is_none());
    }

    #[test]
    fn constructors() {
        let db = db_with(&[("/src/omegaconf.py", "class DictConfig: ...\n")]);
        let env = db.program_environment();

        assert_eq!(display(&db, str_literal(&db, "adam")), "Literal[\"adam\"]");
        assert_eq!(display(&db, int_literal(3)), "Literal[3]");
        // `*` marks an instance of `float` itself, not the `int | float` that a `float`
        // annotation means.
        assert_eq!(display(&db, float_instance(&db, &env)), "float*");
        assert_eq!(display(&db, str_instance(&db, &env)), "str");
        assert_eq!(display(&db, list_of_unknown(&db, &env)), "list[Unknown]");
        assert_eq!(
            display(&db, dict_of_str_unknown(&db, &env)),
            "dict[str, Unknown]"
        );

        let class = member(&db, "omegaconf", &["DictConfig"]).unwrap();
        assert_eq!(
            display(&db, instance_of(&db, &env, class).unwrap()),
            "DictConfig"
        );
        assert!(instance_of(&db, &env, int_literal(1)).is_none());
    }

    #[test]
    fn check_call_reports_argument_type_errors() {
        let db = db_with(&[(
            "/src/lib.py",
            "from typing import Literal\ndef train(lr: float, name: Literal['adam', 'sgd'] = 'adam') -> int: ...\n",
        )]);
        let env = db.program_environment();

        let ok = [
            (Arg::Positional, float_instance(&db, &env)),
            (Arg::Keyword("name"), str_literal(&db, "sgd")),
        ];
        assert_eq!(
            call(&db, "lib", &["train"], &ok),
            ("int".to_string(), vec![], false)
        );

        let bad = [
            (Arg::Keyword("name"), str_literal(&db, "adamw")),
            (Arg::Keyword("lr"), str_literal(&db, "x")),
        ];
        let (_, errors, _) = call(&db, "lib", &["train"], &bad);
        assert_eq!(
            errors,
            [
                "0: name: Literal[\"adam\", \"sgd\"] <- Literal[\"adamw\"]",
                "1: lr: float <- Literal[\"x\"]",
            ]
        );

        // Missing and unknown arguments are not reported.
        let unknown = [(Arg::Keyword("other"), int_literal(1))];
        assert_eq!(
            call(&db, "lib", &["train"], &unknown).1,
            Vec::<String>::new()
        );
    }

    #[test]
    fn check_call_matches_parameters() {
        let db = db_with(&[(
            "/src/lib.py",
            "def f(a: int, *args: str, b: list[int]) -> None: ...\n",
        )]);
        let env = db.program_environment();
        let callable = member(&db, "lib", &["f"]).unwrap();
        let args = [
            (Arg::Positional, int_literal(1)),
            (Arg::Positional, str_literal(&db, "x")),
            (Arg::Keyword("b"), list_of_unknown(&db, &env)),
        ];
        let check = check_call(&db, &env, callable, &args);
        let matched: Vec<_> = check
            .matched_parameters
            .iter()
            .map(|matched| (matched.argument_index, display(&db, matched.annotated_type)))
            .collect();
        assert_eq!(
            matched,
            [
                (0, "int".to_string()),
                (1, "str".to_string()),
                (2, "list[int]".to_string())
            ]
        );
    }

    #[test]
    fn check_call_on_classes() {
        let db = db_with(&[(
            "/src/lib.py",
            "\
from dataclasses import dataclass

class WithInit:
    def __init__(self, size: int) -> None: ...

class WithNew:
    def __new__(cls, size: int) -> 'WithNew': ...

@dataclass
class Data:
    size: int
    name: str = ''
",
        )]);
        let bad = [(Arg::Keyword("size"), str_literal(&db, "x"))];
        for class in ["WithInit", "WithNew", "Data"] {
            let (return_type, errors, _) = call(&db, "lib", &[class], &bad);
            assert_eq!(return_type, class);
            assert_eq!(errors, ["0: size: int <- Literal[\"x\"]"], "{class}");
        }
        let positional = [
            (Arg::Positional, int_literal(1)),
            (Arg::Positional, int_literal(2)),
        ];
        assert_eq!(
            call(&db, "lib", &["Data"], &positional).1,
            ["1: name: str <- Literal[2]"]
        );
    }

    #[test]
    fn check_call_on_overloads() {
        let db = db_with(&[(
            "/src/lib.py",
            "\
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x): ...
",
        )]);
        let env = db.program_environment();
        let int = [(Arg::Positional, int_literal(1))];
        assert_eq!(
            call(&db, "lib", &["f"], &int),
            ("int".to_string(), vec![], false)
        );
        let string = [(Arg::Keyword("x"), str_instance(&db, &env))];
        assert_eq!(
            call(&db, "lib", &["f"], &string),
            ("str".to_string(), vec![], false)
        );
        let float = [(Arg::Positional, float_instance(&db, &env))];
        assert_eq!(
            call(&db, "lib", &["f"], &float),
            ("Unknown".to_string(), vec![], true)
        );
    }

    #[test]
    fn check_call_through_functools_partial() {
        let db = db_with(&[("/src/lib.py", "def f(a: int, b: str) -> float: ...\n")]);
        let env = db.program_environment();
        let partial = member(&db, "functools", &["partial"]).unwrap();
        let f = member(&db, "lib", &["f"]).unwrap();

        let check = check_call(
            &db,
            &env,
            partial,
            &[(Arg::Positional, f), (Arg::Keyword("a"), int_literal(1))],
        );
        assert!(check.errors.is_empty());
        assert_eq!(
            display(&db, check.return_type),
            "partial[(*, a: int = 1, b: str) -> float]"
        );

        let bad = [(Arg::Positional, f), (Arg::Keyword("b"), int_literal(1))];
        let errors = check_call(&db, &env, partial, &bad).errors;
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].argument_index, 1);
        assert_eq!(display(&db, errors[0].expected), "str");
    }

    /// The names of the reachable definitions of `name` in `scope`, as `kind@line`.
    fn reachable(db: &TestDb, scope: ScopeId<'_>, name: &str) -> Vec<String> {
        let module = parsed_module(db, scope.program_file(db).python_file(db)).load(db);
        let source = ruff_db::source::source_text(db, scope.program_file(db).file(db));
        reachable_definitions(db, scope, name)
            .into_iter()
            .map(|definition| {
                let range = definition.kind(db).full_range(&module);
                let line = source[..range.start().to_usize()].matches('\n').count() + 1;
                let kind = match definition.kind(db) {
                    DefinitionKind::Function(_) => "def",
                    DefinitionKind::Class(_) => "class",
                    _ => "other",
                };
                format!("{kind}@{line}")
            })
            .collect()
    }

    const BRANCHES: &str = "\
import sys

if sys.version_info >= (3, 12):
    def f(a): ...
else:
    def f(b): ...

class C:
    if sys.version_info >= (3, 12):
        def m(self): ...
    if sys.platform == 'win32':
        def w(self): ...
    def g(self): ...
    if flag():
        g: int = 1

def flag() -> bool: ...
";

    #[test]
    fn reachable_definitions_follow_python_version() {
        for (version, expected) in [
            (PythonVersion::PY310, "def@6"),
            (PythonVersion::PY313, "def@4"),
        ] {
            let db = TestDbBuilder::new()
                .with_python_version(version)
                .with_file("/src/lib.py", BRANCHES)
                .build()
                .unwrap();
            let file = db.program_file(system_path_to_file(&db, "/src/lib.py").unwrap());
            assert_eq!(reachable(&db, global_scope(&db, file), "f"), [expected]);
            assert_eq!(reachable(&db, global_scope(&db, file), "C"), ["class@8"]);
            assert!(reachable(&db, global_scope(&db, file), "missing").is_empty());
        }
    }

    #[test]
    fn reachable_definitions_in_class_body() {
        let db = TestDbBuilder::new()
            .with_python_version(PythonVersion::PY310)
            .with_python_platform(ty_python_core::platform::PythonPlatform::Identifier(
                "linux".to_string(),
            ))
            .with_file("/src/lib.py", BRANCHES)
            .build()
            .unwrap();
        let file = db.program_file(system_path_to_file(&db, "/src/lib.py").unwrap());
        let module = parsed_module(&db, file.python_file(&db)).load(&db);
        let class = module.suite()[2].as_class_def_stmt().unwrap();
        let scope = semantic_index(&db, file)
            .node_scope(NodeWithScopeRef::Class(class))
            .to_scope_id(&db, file);

        assert!(reachable(&db, scope, "m").is_empty());
        assert!(reachable(&db, scope, "w").is_empty());
        // A condition ty cannot decide keeps its definitions.
        assert_eq!(reachable(&db, scope, "g"), ["def@13", "other@15"]);
    }
}
