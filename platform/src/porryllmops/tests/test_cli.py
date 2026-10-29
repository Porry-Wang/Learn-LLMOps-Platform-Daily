"""S3 CLI 冒烟：四个命令能串成一条线。"""

from porryllmops.cli import main


def test_submit_status_events(tmp_path, capsys):
    db = str(tmp_path / "db.json")

    assert main([
        "--db", db, "submit", "--task", "finetune-7b",
        "--set", "lr=1e-4", "--simulate", "ok", "--gpus", "8",
    ]) == 0
    submitted = capsys.readouterr().out
    assert "run-0001" in submitted
    assert "ckpt=complete" in submitted
    assert "10/10 running" in submitted

    assert main(["--db", db, "status", "run-0001"]) == 0
    assert "succeeded" in capsys.readouterr().out

    assert main(["--db", db, "events", "run-0001"]) == 0
    timeline = capsys.readouterr().out
    assert "run.running" in timeline
    assert "run.succeeded" in timeline


def test_retry_after_trainer_lost(tmp_path, capsys):
    db = str(tmp_path / "db.json")

    assert main([
        "--db", db, "submit", "--task", "t",
        "--simulate", "trainer-lost", "--gpus", "4",
    ]) == 0
    assert "fail_run" in capsys.readouterr().out

    assert main(["--db", db, "retry", "run-0001"]) == 0
    assert "attempt 2" in capsys.readouterr().out

    assert main(["--db", db, "status", "run-0001"]) == 0
    after_retry = capsys.readouterr().out
    assert "pending" in after_retry
    assert "attempt=2" in after_retry


def test_bad_ckpt_is_not_success(tmp_path, capsys):
    db = str(tmp_path / "db.json")

    main([
        "--db", db, "submit", "--task", "t",
        "--simulate", "bad-ckpt", "--gpus", "2",
    ])
    out = capsys.readouterr().out
    assert "ckpt=partial" in out
    assert "可续跑    否" in out
