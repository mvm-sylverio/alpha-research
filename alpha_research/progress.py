"""Optional synchronous observation of research work, without runtime dependencies.

Callbacks receive ``(phase, completed, total)`` at safe work boundaries. Counts
describe processed units, including skipped/undefined units, rather than valid
estimates or overall elapsed-time percentages. ``total=None`` denotes an unknown
count. Exceptions raised by an observer propagate unchanged, allowing callers
to request cooperative cancellation without a library-specific exception type.
Observers should raise ordinary exceptions, not StopIteration: Python converts
StopIteration escaping a generator into RuntimeError with the original cause.
"""

from collections.abc import Callable, Iterable, Iterator, Sized
from inspect import (
    isasyncgen,
    isasyncgenfunction,
    isawaitable,
    iscoroutine,
    iscoroutinefunction,
    isgenerator,
    isgeneratorfunction,
)
from typing import TypeAlias, TypeVar

ProgressCallback: TypeAlias = Callable[[str, int, int | None], None]
_T = TypeVar('_T')

__all__ = ['ProgressCallback']


def _validate_progress_callback(on_progress: ProgressCallback | None) -> None:
    """Validate an optional synchronous observer without invoking it.

    Parameters
    ----------
    on_progress : callable | None
        Optional callback accepting phase, completed count, and total count.

    Returns
    -------
    None
        The callback is supported.

    Raises
    ------
    TypeError
        If the observer is not callable or is an async/generator function.
    """
    if on_progress is None:
        return
    if not callable(on_progress):
        raise TypeError('on_progress must be a synchronous callback or None.')
    implementation = type(on_progress).__call__
    if any(
            iscoroutinefunction(value) or isgeneratorfunction(value) or isasyncgenfunction(value)
            for value in (on_progress, implementation)):
        raise TypeError('on_progress must be a synchronous callback or None.')


def _report_progress(on_progress: ProgressCallback | None, phase: str,
                     completed: int, total: int | None) -> None:
    """Report exact local work counts without catching observer exceptions.

    Parameters
    ----------
    on_progress, phase, completed, total
        Observer, nonempty phase, processed count, and known or unknown total.
        Empty work may report zero completed out of zero total.

    Returns
    -------
    None
        The optional observer has received the event.

    Raises
    ------
    TypeError, ValueError, Exception
        If the callback/counts are invalid or the observer raises an exception.
    """
    _validate_progress_callback(on_progress)
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError('Progress phase must be non-empty.')
    if type(completed) is not int or completed < 0 or total is not None and (
            type(total) is not int or total < completed):
        raise ValueError('Progress counts must satisfy 0 <= completed <= total.')
    if on_progress is not None:
        observation = on_progress(phase, completed, total)
        if isawaitable(observation) or isgenerator(observation) or isasyncgen(observation):
            # Close unstarted native coroutines/generators without running them.
            if iscoroutine(observation) or isgenerator(observation):
                observation.close()
            raise TypeError('on_progress must complete synchronously.')


def _progress_iter(values: Iterable[_T], on_progress: ProgressCallback | None,
                   phase: str, *, total: int | None = None) -> Iterator[_T]:
    """Observe lazy work units before generation and after their consumer resumes.

    Parameters
    ----------
    values, on_progress, phase, total
        Original iterable, optional observer, local phase, and optional count.
        Sized iterables provide their count. Closing or breaking the iterator
        does not assert that its suspended unit completed.

    Returns
    -------
    iterator
        Unchanged values in their original order. No callback uses the original
        iteration path without calculating extra group counts or buffering data.

    Raises
    ------
    TypeError, ValueError, Exception
        If the observer/count is invalid, iteration fails, or observation stops.
    RuntimeError
        If a callback raises StopIteration inside this generator (Python's
        generator safety rule); the original exception remains its cause.
    """
    _validate_progress_callback(on_progress)
    _report_progress(None, phase, 0, total)
    if on_progress is None:
        yield from values
        return
    if total is None and isinstance(values, Sized):
        total = len(values)
    _report_progress(on_progress, phase, 0, total)
    iterator = iter(values)
    completed = 0
    while True:
        # Repeating the current count checks cancellation before allocating the
        # next unit and after generation, before its statistical calculation.
        _report_progress(on_progress, phase, completed, total)
        try:
            value = next(iterator)
        except StopIteration:
            if total is not None and completed != total:
                raise ValueError('Work unit count differs from the declared total.')
            return
        if total is not None and completed >= total:
            raise ValueError('Work unit count exceeds the declared total.')
        _report_progress(on_progress, phase, completed, total)
        yield value
        completed += 1
        _report_progress(on_progress, phase, completed, total)


def _scope_progress(on_progress: ProgressCallback | None, scope: str) -> ProgressCallback | None:
    """Prefix nested observation phases without changing the callback contract.

    Parameters
    ----------
    on_progress, scope
        Original observer and nonempty scope such as a feature or horizon.

    Returns
    -------
    callable | None
        Scoped callback, or None to retain the unobserved execution path.

    Raises
    ------
    TypeError, ValueError, Exception
        If the callback/scope is invalid or the original observer later fails.
    """
    _validate_progress_callback(on_progress)
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError('Progress scope must be non-empty.')
    if on_progress is None:
        return None

    def scoped(phase: str, completed: int, total: int | None) -> None:
        """Forward one nested work event to the original observer.

        Parameters
        ----------
        phase, completed, total
            Local phase and exact work counts from the nested operation.

        Returns
        -------
        None
            The original observer has received the prefixed event.

        Raises
        ------
        Exception
            Propagates observation failures unchanged.
        """
        _report_progress(on_progress, f'{scope}/{phase}', completed, total)

    return scoped


def _progress_kwargs(on_progress: ProgressCallback | None, scope: str | None = None) -> dict:
    """Forward observation only when enabled, preserving unobserved call arguments.

    Parameters
    ----------
    on_progress, scope
        Observer and optional nested phase prefix.

    Returns
    -------
    dict
        Optional on_progress keyword, or an empty mapping for existing calls.

    Raises
    ------
    TypeError, ValueError
        If the observer or optional scope is invalid.
    """
    _validate_progress_callback(on_progress)
    if scope is not None:
        on_progress = _scope_progress(on_progress, scope)
    return {} if on_progress is None else {'on_progress': on_progress}
