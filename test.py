from ClassApplication import *
import pytest
import io, sys

# Testing user inputs for all (8) attributes/details of the loan application
@pytest.mark.parametrize('input, district', [('cenTER','Center'),('SOUTH','South'), ('north','North')]) # Testing valid inputs
def test_district(capfd, monkeypatch, input, district):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
    set_district()
    output = capfd.readouterr().out
    assert district in output

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#']) # Testing invalid inputs
def test_set_district_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)

@pytest.mark.parametrize('input, gender', [('f','F'),('m','M')])
def test_set_gender(capfd, monkeypatch, input, gender):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
    set_gender()
    output = capfd.readouterr().out
    assert gender in output

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#'])
def test_set_gender_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)

@pytest.mark.parametrize('input, ownership', [('oNE OWner','One Owner' ), ('ONe oWner', 'One Owner'),
                                              ('multiple owners','Multiple Owners'), ('MULTIPLE OWNERS','Multiple Owners')])
def test_set_ownership(capfd, monkeypatch, input, ownership):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
    set_ownership()
    output = capfd.readouterr().out
    assert ownership in output

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#'])
def test_set_ownership_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#'])
def test_set_age_range_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)

@pytest.mark.parametrize('input, country', [('GERMANY','Germany' ), ('italy', 'Italy'),('FrAnCe','France')])
def test_set_country(capfd, monkeypatch, input, country):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
    set_country()
    output = capfd.readouterr().out
    assert country in output

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#'])
def test_set_country_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)

@pytest.mark.parametrize('input, business_status', [('new BUSINESS','New Business' ), ('NEW business', 'New Business'),
                                                    ('Old business','Old Business')])
def test_set_business_status(capfd, monkeypatch, input, business_status):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
    set_business_status()
    output = capfd.readouterr().out
    assert business_status in output

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#'])
def test_set_business_status_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)

@pytest.mark.parametrize('input, field', [('retail','Retail' ), ('marketing', 'Marketing'),('ATTRACTIONS','Attractions')])
def test_set_field(capfd, monkeypatch, input, field):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
    set_field()
    output = capfd.readouterr().out
    assert field in output

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#'])
def test_set_field_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)

@pytest.mark.parametrize('input, consultant', [('a.s','A.S' ), ('T.z', 'T.Z')])
def test_set_consultant(capfd, monkeypatch, input, consultant):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
    set_consultant()
    output = capfd.readouterr().out
    assert consultant in output

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#'])
def test_set_consultant_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)

@pytest.mark.parametrize('input, loan_amount_range', [('very high','Very High' ), ('MEDIUM', 'Medium')])
def test_set_loan_amount_range(capfd, monkeypatch, input, loan_amount_range):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
    set_loan_amount_range()
    output = capfd.readouterr().out
    assert loan_amount_range in output

@pytest.mark.parametrize('invalid_input', ['123', 'abc', '!@#'])
def test_set_loan_amount_rang_error_raised(capfd, monkeypatch, invalid_input):
    with pytest.raises(AssertionError) as e:
        assert "NOT A VALID INPUT . PLEASE TRY AGAIN" in str(e.value)


