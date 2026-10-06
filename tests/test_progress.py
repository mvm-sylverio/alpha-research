"""Contract tests using synthetic work, independent from numerical kernels."""

from functools import partial

import pytest

from alpha_research.progress import (
    _progress_iter,
    _progress_kwargs,
    _report_progress,
    _scope_progress,
    _validate_progress_callback,
)


@pytest.mark.parametrize('callback', [None, lambda *args: None, print])
def test_validate_callback_accepts_synchronous_observers(callback):
    """Should validate an optional synchronous observer without calling it."""
    _validate_progress_callback(callback)


@pytest.mark.parametrize('callback', [False, 12, 'observer', [], {}])
def test_validate_callback_rejects_noncallables(callback):
    """Should reject unsupported observers before work starts."""
    with pytest.raises(TypeError, match='synchronous'):
        _validate_progress_callback(callback)


def test_validate_callback_rejects_async_and_generator_forms():
    """Should reject functions, partials, and callable objects that cannot observe synchronously."""
    async def asynchronous(*args):
        """Should represent an unsupported coroutine observer."""

    def generator(*args):
        """Should represent an unsupported lazy observer."""
        yield args

    async def async_generator(*args):
        """Should represent an unsupported asynchronous generator observer."""
        yield args

    class AsyncObserver:
        """Represent an unsupported asynchronous callable object."""

        __call__ = asynchronous

    class GeneratorObserver:
        """Represent an unsupported generator callable object."""

        __call__ = generator

    for callback in (asynchronous, generator, async_generator, partial(async_generator), partial(asynchronous), partial(generator),
                     AsyncObserver(), GeneratorObserver()):
        with pytest.raises(TypeError, match='synchronous'):
            _validate_progress_callback(callback)


@pytest.mark.parametrize('phase,completed,total', [
    ('', 0, 1), (' ', 0, 1), (None, 0, 1), ('work', -1, 1),
    ('work', True, 1), ('work', 0, False), ('work', 1.5, 2),
    ('work', 0, '2'), ('work', 2, 1), ('work', 0, -1),
])
def test_report_rejects_invalid_events_even_without_observer(phase, completed, total):
    """Should reject malformed events independently of observation being enabled."""
    with pytest.raises(ValueError):
        _report_progress(None, phase, completed, total)


@pytest.mark.parametrize('completed,total', [(0, 0), (0, None), (3, None), (2, 2)])
def test_report_preserves_exact_counts(completed, total):
    """Should preserve unknown and empty work counts without fabricating percentages."""
    events = []
    _report_progress(lambda *event: events.append(event), 'work', completed, total)
    assert events == [('work', completed, total)]


@pytest.mark.parametrize('exception', [ValueError('stop'), TypeError('stop'), RuntimeError('stop')])
def test_report_propagates_original_exception(exception):
    """Should propagate the same exception instance without reclassification."""
    callback = lambda *event: (_ for _ in ()).throw(exception)
    with pytest.raises(type(exception)) as caught:
        _report_progress(callback, 'work', 0, 1)
    assert caught.value is exception


def test_report_rejects_lazy_results_hidden_by_synchronous_wrappers():
    """Should reject unexecuted coroutine/generator results without silently dropping observation."""
    observed = []
    async def asynchronous(*args):
        """Should stay unexecuted when hidden by an unsupported wrapper."""
        observed.append(args)
    for callback in (lambda *args: asynchronous(*args), lambda *args: (value for value in args)):
        with pytest.raises(TypeError, match='synchronously'):
            _report_progress(callback, 'work', 0, 1)
    assert observed == []


def test_iterator_is_lazy_and_counts_only_processed_work():
    """Should avoid prefetch and count completion only when the consumer resumes."""
    generated = []
    events = []
    values = (generated.append(index) or index for index in range(3))
    iterator = _progress_iter(values, lambda *event: events.append(event), 'work', total=3)
    assert generated == events == []
    assert next(iterator) == 0
    assert generated == [0]
    assert all(event[1] == 0 for event in events)
    assert next(iterator) == 1
    assert generated == [0, 1]
    iterator.close()
    assert max(event[1] for event in events) == 1


def test_iterator_can_stop_before_generating_any_work():
    """Should propagate cancellation before the input iterator consumes a unit."""
    generated = []
    exception = RuntimeError('stop')
    values = (generated.append(index) for index in range(5))
    with pytest.raises(RuntimeError) as caught:
        list(_progress_iter(values, lambda *event: (_ for _ in ()).throw(exception), 'work'))
    assert caught.value is exception
    assert generated == []


def test_iterator_can_stop_after_generation_before_calculation():
    """Should check an allocated unit before handing it to the statistical consumer."""
    generated = []
    consumed = []
    exception = ValueError('stop after allocation')
    callback = lambda *event: (_ for _ in ()).throw(exception) if generated else None
    values = (generated.append(index) or index for index in range(3))
    with pytest.raises(ValueError) as caught:
        consumed.extend(_progress_iter(values, callback, 'work', total=3))
    assert caught.value is exception
    assert generated == [0]
    assert consumed == []


@pytest.mark.parametrize('values,total', [([1, 2], 1), ([1], 2)])
def test_iterator_rejects_incorrect_declared_totals(values, total):
    """Should detect both overflow and underflow of a declared work count."""
    with pytest.raises(ValueError, match='count'):
        list(_progress_iter(values, lambda *event: None, 'work', total=total))


def test_iterator_reports_unknown_and_empty_totals():
    """Should handle empty and unsized iterables without buffering or fake totals."""
    events = []
    assert list(_progress_iter(iter([10, 20]), lambda *event: events.append(event), 'work')) == [10, 20]
    assert events[-1] == ('work', 2, None)
    events.clear()
    assert list(_progress_iter([], lambda *event: events.append(event), 'empty')) == []
    assert events[-1] == ('empty', 0, 0)
    assert list(_progress_iter(iter([10, 20]), None, 'work')) == [10, 20]


def test_iterator_preserves_input_errors():
    """Should preserve a source iterator failure without reporting the failed unit complete."""
    exception = RuntimeError('input failed')
    events = []
    values = (value if value == 1 else (_ for _ in ()).throw(exception) for value in [1, 2])
    with pytest.raises(RuntimeError) as caught:
        list(_progress_iter(values, lambda *event: events.append(event), 'work', total=2))
    assert caught.value is exception
    assert max(event[1] for event in events) == 1


def test_iterator_never_treats_observer_stopiteration_as_success():
    """Should fail rather than silently truncate work when an observer misuses the iterator protocol."""
    exception = StopIteration('invalid cancellation exception')
    consumed = []
    # Throw directly: a generator-expression throw would already apply PEP 479.
    def observe(*event):
        """Should expose an invalid StopIteration raised by a synchronous observer."""
        raise exception
    with pytest.raises(RuntimeError) as caught:
        consumed.extend(_progress_iter([1, 2], observe, 'work'))
    assert caught.value.__cause__ is exception
    assert consumed == []


def test_scope_and_forwarding_preserve_callback_and_noop_arguments():
    """Should compose nested phases and omit the new keyword when observation is disabled."""
    events = []
    callback = lambda *event: events.append(event)
    assert _scope_progress(None, 'feature/x') is None
    assert _progress_kwargs(None) == _progress_kwargs(None, 'feature/x') == {}
    assert _progress_kwargs(callback)['on_progress'] is callback
    scoped = _progress_kwargs(_scope_progress(callback, 'feature/x'), 'horizon/3')['on_progress']
    scoped('dates', 2, 5)
    assert events == [('feature/x/horizon/3/dates', 2, 5)]


@pytest.mark.parametrize('scope', ['', ' ', None, 3])
def test_scope_rejects_invalid_prefix(scope):
    """Should validate explicit scope names even without an observer."""
    with pytest.raises(ValueError):
        _scope_progress(None, scope)
    if scope is not None:
        with pytest.raises(ValueError):
            _progress_kwargs(None, scope)


def test_scope_propagates_validation_and_observer_failures():
    """Should validate forwarded events and preserve their original observer exceptions."""
    with pytest.raises(TypeError):
        _progress_kwargs(12)
    with pytest.raises(TypeError):
        _scope_progress(12, 'work')
    exception = TypeError('observer failed')
    callback = _scope_progress(lambda *event: (_ for _ in ()).throw(exception), 'work')
    with pytest.raises(ValueError):
        callback('dates', 5, 2)
    with pytest.raises(TypeError) as caught:
        callback('dates', 0, 2)
    assert caught.value is exception
